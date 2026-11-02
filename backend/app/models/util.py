"""Identity and timestamp helpers for persisted rows."""

from datetime import UTC, datetime
from uuid import uuid4


def new_id() -> str:
    return str(uuid4())


def utcnow() -> datetime:
    return datetime.now(UTC)


def elapsed_seconds(started: datetime, finished: datetime) -> float:
    """Seconds between two clocks.

    SQLite returns a stored timestamp with no timezone. A value just assigned
    in this process still carries UTC, and subtracting the two types raises.
    """
    return (_as_utc(finished) - _as_utc(started)).total_seconds()


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
