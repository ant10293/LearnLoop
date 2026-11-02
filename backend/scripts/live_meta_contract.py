"""Paid, explicitly invoked check of the Meta output contracts. Never part of the test suite."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline.contracts import (  # noqa: E402
    DISCOVERY,
    INTERPRETATION,
    SINGLE_PASS,
    StructuredOutputError,
    detect_runaway,
    validate_output,
    validate_strict_schema,
)
from app.pipeline.prompts import DISCOVERY as DISCOVERY_PROMPT  # noqa: E402
from app.pipeline.prompts import INTERPRETATION as INTERPRETATION_PROMPT  # noqa: E402
from app.pipeline.prompts import compose  # noqa: E402
from app.providers.meta_provider import MetaProvider  # noqa: E402
from app.providers.types import GenerationRequest  # noqa: E402

SMALL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer"],
    "properties": {"answer": {"type": "string"}},
}


def _secret() -> str:
    secret = os.environ.get("META_API_KEY", "")
    if secret:
        return secret
    from app.db import SessionLocal
    from app.services.credentials import load_secret

    session = SessionLocal()
    try:
        return load_secret(session, "meta")[0]
    finally:
        session.close()


async def _one(provider, secret, args, name, schema, system, contract=None) -> dict:
    video = Path(args.video) if contract is not None else None
    request = GenerationRequest(
        model_id=args.model,
        system_prompt=system,
        user_prompt="Use only the attached evidence." if video else "Reply with the word ready.",
        video_path=video,
        modality="native_video" if video else "sampled_frames",
        json_schema=schema,
        schema_name=name,
        timeout_s=1800,
    )
    report = {"check": name, "accepted": False, "completed": False, "json": False, "contract": False}
    try:
        response = await provider.generate(secret, request)
    except Exception as exc:
        return {**report, "error": str(exc)[:500]}
    report |= {
        "accepted": True,
        "completed": not response.incomplete_reason,
        "incomplete_reason": response.incomplete_reason,
        "output_tokens": response.usage.output_tokens,
        "raw_output_chars": len(response.text),
        "runaway": detect_runaway(response.text),
    }
    try:
        json.loads(response.text)
        report["json"] = True
    except json.JSONDecodeError:
        pass
    if contract is None:
        report["contract"] = report["json"] and set(json.loads(response.text)) == {"answer"}
        return report
    try:
        validate_output(contract, response.text, incomplete_reason=response.incomplete_reason)
        report["contract"] = True
    except StructuredOutputError as exc:
        report |= {"violation": exc.code, "detail": str(exc)[:300]}
    return report


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="Confirm that this spends real tokens.")
    parser.add_argument("--model", default="muse-wonder-1.0")
    parser.add_argument("--video", required=True)
    parser.add_argument("--mode", default="fixed_camera")
    args = parser.parse_args()
    if not args.yes:
        print("This sends paid requests to Meta. Pass --yes to run it.")
        return 2
    secret = _secret()
    if not secret:
        print("No Meta API key is configured.")
        return 2
    validate_strict_schema(SMALL_SCHEMA)
    scene = compose("scene", args.mode)
    full = "\n\n".join(compose(key, args.mode) for key in ("scene", "actions", "states", "workflow", "ambiguity"))
    meaning = "\n\n".join(compose(key, args.mode) for key in ("actions", "states", "workflow", "ambiguity"))
    provider = MetaProvider()
    reports = []
    for name, schema, system, contract in (
        ("small_strict", SMALL_SCHEMA, "Return JSON only.", None),
        (DISCOVERY.name, DISCOVERY.schema, f"{scene}\n\n{DISCOVERY_PROMPT}", DISCOVERY),
        (INTERPRETATION.name, INTERPRETATION.schema, f"{meaning}\n\n{INTERPRETATION_PROMPT}", INTERPRETATION),
        (SINGLE_PASS.name, SINGLE_PASS.schema, full, SINGLE_PASS),
    ):
        report = await _one(provider, secret, args, name, schema, system, contract)
        reports.append(report)
        print(json.dumps(report), flush=True)
    return 0 if all(item["contract"] for item in reports) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
