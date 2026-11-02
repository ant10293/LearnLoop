"""Add task-project columns on databases created before those fields existed."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine

COLUMN_UPGRADES = {
    "projects": [("is_task", "BOOLEAN DEFAULT 0")],
    "video_assets": [
        ("session_id", "VARCHAR(36)"),
        ("research_mode", "VARCHAR(64) DEFAULT ''"),
        ("capture_time_raw", "VARCHAR(120) DEFAULT ''"),
        ("capture_time_utc", "VARCHAR(40) DEFAULT ''"),
        ("capture_timezone", "VARCHAR(40) DEFAULT ''"),
        ("timestamp_source", "VARCHAR(40) DEFAULT 'absent'"),
        ("timestamp_reliability", "VARCHAR(32) DEFAULT 'absent'"),
        ("timestamp_note", "TEXT DEFAULT ''"),
        ("file_mtime", "VARCHAR(40) DEFAULT ''"),
        ("order_index", "FLOAT"),
        ("order_locked", "BOOLEAN DEFAULT 0"),
        ("order_method", "VARCHAR(32) DEFAULT ''"),
        ("order_ambiguous", "BOOLEAN DEFAULT 0"),
        ("audio_probed", "BOOLEAN DEFAULT 0"),
        ("has_audio", "BOOLEAN DEFAULT 0"),
        ("audio_codec", "VARCHAR(80) DEFAULT ''"),
        ("audio_channels", "INTEGER DEFAULT 0"),
        ("ingest_status", "VARCHAR(32) DEFAULT 'ready'"),
        ("ingest_progress", "FLOAT DEFAULT 100"),
        ("ingest_detail", "VARCHAR(200) DEFAULT ''"),
        ("ingest_error", "TEXT DEFAULT ''"),
    ],
    "analysis_runs": [("starred", "BOOLEAN NOT NULL DEFAULT 0")],
    "model_responses": [("price_known", "BOOLEAN NOT NULL DEFAULT 0")],
    "spatial_track_points": [
        ("track_status", "VARCHAR(32) DEFAULT ''"),
        ("association_score", "FLOAT"),
        ("detector_id", "VARCHAR(80) DEFAULT ''"),
    ],
    "object_observations": [
        ("confidence", "FLOAT"),
        ("association_status", "VARCHAR(32) DEFAULT ''"),
        ("source_model", "VARCHAR(80) DEFAULT ''"),
        ("request_id", "VARCHAR(120) DEFAULT ''"),
        ("mask_ref", "VARCHAR(200) DEFAULT ''"),
    ],
}


def ensure_task_columns(engine: Engine) -> None:
    if engine.dialect.name != "sqlite":
        return
    with engine.begin() as conn:
        tables = {
            row[0]
            for row in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        }
        for table, columns in COLUMN_UPGRADES.items():
            if table not in tables:
                continue
            present = {row[1] for row in conn.execute(text(f"PRAGMA table_info({table})"))}
            for name, ddl in columns:
                if name not in present:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))
        if "model_responses" in tables:
            present = {row[1] for row in conn.execute(text("PRAGMA table_info(model_responses)"))}
            if "price_known" in present:
                conn.execute(
                    text("UPDATE model_responses SET price_known = 1 WHERE cost_usd > 0 AND price_known = 0")
                )
    allow_missing_project(engine, "video_assets")
    allow_missing_project(engine, "analysis_runs")
    unique_manual_models(engine)


def unique_manual_models(engine: Engine) -> None:
    """Drop repeated manual model ids, keeping the earliest, then enforce one row per provider and id."""
    if engine.dialect.name != "sqlite":
        return
    with engine.begin() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM sqlite_master WHERE type='table' AND name='manual_models'")
        ).first()
        if not exists:
            return
        conn.execute(text(DEDUPE_MANUAL_MODELS))
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_manual_model_provider_model "
                "ON manual_models (provider_id, model_id)"
            )
        )


DEDUPE_MANUAL_MODELS = """
DELETE FROM manual_models
WHERE id NOT IN (
    SELECT id FROM (
        SELECT id, ROW_NUMBER() OVER (
            PARTITION BY provider_id, model_id ORDER BY created_at, id
        ) AS position
        FROM manual_models
    ) AS ranked WHERE position = 1
)
"""


def allow_missing_project(engine: Engine, table: str) -> None:
    """Let project_id be empty. SQLite cannot drop NOT NULL without rebuilding the table."""
    if engine.dialect.name != "sqlite":
        return
    with engine.connect() as conn:
        rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
        column = next((row for row in rows if row[1] == "project_id"), None)
        if column is None or column[3] == 0:
            return
        conn.commit()
        conn.exec_driver_sql("PRAGMA foreign_keys=OFF")
        try:
            _rebuild_without_not_null(conn, table, "project_id", rows)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.exec_driver_sql("PRAGMA foreign_keys=ON")
            conn.commit()


def _rebuild_without_not_null(conn, table: str, column: str, info: list) -> None:
    foreign_keys = conn.execute(text(f"PRAGMA foreign_key_list({table})")).fetchall()
    indexes = conn.execute(text(f"PRAGMA index_list({table})")).fetchall()
    index_columns: list[tuple[str, bool, list[str]]] = []
    for index in indexes:
        origin = index[3] if len(index) > 3 else ""
        partial = index[4] if len(index) > 4 else 0
        if origin == "pk" or partial or str(index[1]).startswith("sqlite_"):
            continue
        cols = [
            col[2]
            for col in conn.execute(text(f'PRAGMA index_info("{index[1]}")')).fetchall()
            if col[2]
        ]
        if cols:
            index_columns.append((index[1], bool(index[2]), cols))
    definitions = [_column_sql(row, column) for row in info]
    definitions.extend(_foreign_key_sql(foreign_keys))
    temporary = f"{table}__project_nullable"
    names = ", ".join(f'"{row[1]}"' for row in info)
    conn.exec_driver_sql(f'CREATE TABLE "{temporary}" ({", ".join(definitions)})')
    conn.exec_driver_sql(f'INSERT INTO "{temporary}" ({names}) SELECT {names} FROM "{table}"')
    conn.exec_driver_sql(f'DROP TABLE "{table}"')
    conn.exec_driver_sql(f'ALTER TABLE "{temporary}" RENAME TO "{table}"')
    for name, unique, cols in index_columns:
        kind = "UNIQUE INDEX" if unique else "INDEX"
        quoted = ", ".join(f'"{col}"' for col in cols)
        conn.exec_driver_sql(f'CREATE {kind} IF NOT EXISTS "{name}" ON "{table}" ({quoted})')


def _column_sql(row, nullable_name: str) -> str:
    _cid, name, coltype, notnull, default, primary = row
    if name == nullable_name:
        notnull = 0
    parts = [f'"{name}" {coltype or "TEXT"}']
    if primary:
        parts.append("PRIMARY KEY")
    elif notnull:
        parts.append("NOT NULL")
    if default is not None:
        parts.append(f"DEFAULT {default}")
    return " ".join(parts)


def _foreign_key_sql(rows: list) -> list[str]:
    grouped: dict[int, list] = {}
    for row in rows:
        grouped.setdefault(row[0], []).append(row)
    clauses = []
    for group in grouped.values():
        group.sort(key=lambda row: row[1])
        columns = ", ".join(f'"{row[3]}"' for row in group)
        targets = ", ".join(f'"{row[4]}"' for row in group)
        clause = f'FOREIGN KEY ({columns}) REFERENCES "{group[0][2]}" ({targets})'
        on_update, on_delete = group[0][5], group[0][6]
        if on_delete and str(on_delete).upper() != "NO ACTION":
            clause += f" ON DELETE {on_delete}"
        if on_update and str(on_update).upper() != "NO ACTION":
            clause += f" ON UPDATE {on_update}"
        clauses.append(clause)
    return clauses
