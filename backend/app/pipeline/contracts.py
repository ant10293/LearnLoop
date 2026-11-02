"""Strict output contracts for Meta analysis calls.

Meta constrains output to the schema it is sent, and `strict` only makes the
server check that schema. A schema that allows extra properties therefore allows
a model to invent keys until its token budget ends. These contracts close every
object, and each response is validated against the contract it was sent with.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from app.pipeline.parsing import ParseFailure
from app.providers.types import UnsupportedMode
from app.schemas.analysis import AnalysisDocument
from app.schemas.generation import (
    ActionsOutput,
    AmbiguitiesOutput,
    DiscoveryOutput,
    InterpretationOutput,
    ReinspectionOutput,
    SinglePassOutput,
    StatesOutput,
    WorkflowOutput,
)

CONTRACT_VERSION = 1

# Documented limits, checked before a request is sent.
# https://dev.meta.ai/docs/structured-output
MAX_DEPTH = 10
MAX_PROPERTIES = 5_000
MAX_STRING_CHARS = 120_000
MAX_ENUM_VALUES = 1_000
MAX_EXPANDED_NODES = 200_000

_ALLOWED_KEYWORDS = {
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "anyOf",
    "enum",
    "const",
    "$ref",
    "$defs",
    "definitions",
    "description",
}
# Pydantic emits these. The documented subset does not list them. Titles and
# defaults carry no constraint, and bounds are enforced by the canonical models.
_DROPPED_KEYWORDS = {"title", "default", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}


class SchemaContractError(UnsupportedMode):
    """A schema cannot be sent as a strict contract. It stops the run without a retry."""


class StructuredOutputError(ParseFailure):
    """A response broke its contract. Retrying spends the same budget, so the job does not retry."""

    def __init__(self, code: str, message: str, raw_text: str = "", diagnostics: dict | None = None):
        super().__init__(message, raw_text)
        self.code = code
        self.diagnostics = diagnostics or {}


def validate_strict_schema(schema: dict) -> None:
    """Raise when a schema falls outside Meta's strict subset or its size limits."""
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise SchemaContractError("The root of a strict schema must be an object.")
    for keyword in ("anyOf", "oneOf", "allOf", "enum", "not"):
        if keyword in schema:
            raise SchemaContractError(f"The root of a strict schema cannot use {keyword}.")
    definitions = {**schema.get("$defs", {}), **schema.get("definitions", {})}
    tally = {"properties": 0, "strings": 0, "enums": 0}
    _check_node(schema, "#", tally)
    for name, definition in definitions.items():
        tally["strings"] += len(name)
        _check_node(definition, f"#/$defs/{name}", tally)
    if tally["properties"] > MAX_PROPERTIES:
        raise SchemaContractError(f"The schema has {tally['properties']} properties. The limit is {MAX_PROPERTIES}.")
    if tally["strings"] > MAX_STRING_CHARS:
        raise SchemaContractError(f"The schema has {tally['strings']} characters. The limit is {MAX_STRING_CHARS}.")
    if tally["enums"] > MAX_ENUM_VALUES:
        raise SchemaContractError(f"The schema has {tally['enums']} enum values. The limit is {MAX_ENUM_VALUES}.")
    depth, nodes = _expanded_size(schema, definitions, ())
    if depth > MAX_DEPTH:
        raise SchemaContractError(f"The schema nests {depth} levels. The limit is {MAX_DEPTH}.")
    if nodes > MAX_EXPANDED_NODES:
        raise SchemaContractError(f"The schema expands to {nodes} nodes. The limit is {MAX_EXPANDED_NODES}.")


def _check_node(node: Any, path: str, tally: dict[str, int]) -> None:
    if not isinstance(node, dict):
        raise SchemaContractError(f"{path} is not a schema object.")
    unknown = set(node) - _ALLOWED_KEYWORDS
    if unknown:
        raise SchemaContractError(f"{path} uses unsupported keywords: {', '.join(sorted(unknown))}.")
    enum = node.get("enum")
    if enum is not None:
        tally["enums"] += len(enum)
        tally["strings"] += sum(len(str(item)) for item in enum)
    if isinstance(node.get("const"), str):
        tally["strings"] += len(node["const"])
    if node.get("type") == "object":
        properties = node.get("properties")
        if properties is None:
            raise SchemaContractError(f"{path} is an object with no declared properties.")
        if node.get("additionalProperties") is not False:
            raise SchemaContractError(f"{path} must set additionalProperties to false.")
        missing = set(properties) - set(node.get("required", []))
        if missing:
            raise SchemaContractError(f"{path} must require every property. Missing: {', '.join(sorted(missing))}.")
        tally["properties"] += len(properties)
        for name, child in properties.items():
            tally["strings"] += len(name)
            _check_node(child, f"{path}.{name}", tally)
    if "items" in node:
        _check_node(node["items"], f"{path}[]", tally)
    for index, option in enumerate(node.get("anyOf", [])):
        _check_node(option, f"{path}|{index}", tally)


def _expanded_size(node: dict, definitions: dict, seen: tuple[str, ...]) -> tuple[int, int]:
    reference = node.get("$ref")
    if reference is not None:
        name = str(reference).rsplit("/", 1)[-1]
        if name in seen:
            raise SchemaContractError(f"The schema is recursive through {name}. Flatten it to a fixed depth.")
        if name not in definitions:
            raise SchemaContractError(f"The schema references {name}, which is not defined.")
        return _expanded_size(definitions[name], definitions, (*seen, name))
    children = list(node.get("properties", {}).values()) + node.get("anyOf", [])
    if "items" in node:
        children.append(node["items"])
    depth, nodes = 0, 1
    for child in children:
        child_depth, child_nodes = _expanded_size(child, definitions, seen)
        depth = max(depth, child_depth)
        nodes += child_nodes
    nests = node.get("type") in {"object", "array"}
    return depth + (1 if nests else 0), nodes


def compile_strict_schema(model: type[BaseModel]) -> dict:
    """The model's schema without annotations, validated. It is never widened to fit."""
    schema = _without_annotations(model.model_json_schema())
    validate_strict_schema(schema)
    return schema


def _without_annotations(node: Any, *, naming: bool = False) -> Any:
    if isinstance(node, list):
        return [_without_annotations(item) for item in node]
    if not isinstance(node, dict):
        return node
    if naming:
        return {name: _without_annotations(child) for name, child in node.items()}
    cleaned = {}
    for key, value in node.items():
        if key in _DROPPED_KEYWORDS:
            continue
        cleaned[key] = _without_annotations(value, naming=key in {"properties", "$defs", "definitions"})
    return cleaned


def schema_hash(schema: dict) -> str:
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class OutputContract:
    name: str
    model: type[BaseModel]
    schema: dict
    version: int = CONTRACT_VERSION

    @property
    def hash(self) -> str:
        return schema_hash(self.schema)


def _contract(name: str, model: type[BaseModel]) -> OutputContract:
    return OutputContract(name, model, compile_strict_schema(model))


DISCOVERY = _contract("learnloop_discovery", DiscoveryOutput)
INTERPRETATION = _contract("learnloop_interpretation", InterpretationOutput)
SINGLE_PASS = _contract("learnloop_single_pass", SinglePassOutput)
_STAGE_CONTRACTS = {
    "scene_analysis": DISCOVERY,
    "action_analysis": _contract("learnloop_actions", ActionsOutput),
    "state_tracking": _contract("learnloop_states", StatesOutput),
    "workflow": _contract("learnloop_workflow", WorkflowOutput),
    "ambiguity": _contract("learnloop_ambiguities", AmbiguitiesOutput),
    "reinspection": _contract("learnloop_reinspection", ReinspectionOutput),
}
ALL_CONTRACTS = (DISCOVERY, INTERPRETATION, SINGLE_PASS, *_STAGE_CONTRACTS.values())


def contract_for(provider_id: str, stage: str, purpose: str, analysis_pass: str | None) -> OutputContract | None:
    """The contract for one call. Only Meta has them. A Meta stage without one is an error."""
    if provider_id != "meta":
        return None
    if purpose == "discovery":
        return DISCOVERY
    if purpose == "interpretation":
        return INTERPRETATION
    if stage == "scene_analysis" and analysis_pass == "single":
        return SINGLE_PASS
    contract = _STAGE_CONTRACTS.get(stage)
    if contract is None:
        raise SchemaContractError(f"Stage {stage} has no output contract. It was not sent unconstrained.")
    return contract


_EMPTY_FIELD = re.compile(r'"([A-Za-z][A-Za-z0-9]*)[_A-Za-z0-9]*"\s*:\s*(?:\[\s*\]|null|\{\s*\})')
RUNAWAY_MIN_KEYS = 25


def detect_runaway(text: str) -> dict:
    """Name a stem repeated across many distinct empty keys, such as relationships_*.

    This is a diagnostic for a stored answer. It does not repair anything.
    """
    stems: dict[str, dict] = {}
    for match in _EMPTY_FIELD.finditer(text):
        key = match.group(0).split('"')[1]
        stem = match.group(1)
        if key == stem:
            continue
        entry = stems.setdefault(stem, {"keys": set(), "offset": match.start()})
        entry["keys"].add(key)
    if not stems:
        return {"detected": False}
    stem, entry = max(stems.items(), key=lambda item: len(item[1]["keys"]))
    if len(entry["keys"]) < RUNAWAY_MIN_KEYS:
        return {"detected": False}
    return {
        "detected": True,
        "stem": stem,
        "distinct_keys": len(entry["keys"]),
        "first_offset": entry["offset"],
        "examples": sorted(entry["keys"])[:5],
    }


def validate_output(
    contract: OutputContract,
    text: str,
    *,
    incomplete_reason: str = "",
    label: str = "",
) -> AnalysisDocument:
    """Accept only a finished answer that matches its contract exactly. Nothing is repaired."""
    runaway = detect_runaway(text)
    base = {"raw_output_chars": len(text), "runaway": runaway}
    prefix = f"{label}: " if label else ""
    suffix = f" Repeated empty keys under {runaway['stem']}_*." if runaway["detected"] else ""
    if incomplete_reason:
        raise StructuredOutputError(
            "incomplete_output",
            f"{prefix}the answer stopped before it finished ({incomplete_reason}).{suffix}",
            text,
            {**base, "incomplete_reason": incomplete_reason},
        )
    try:
        parsed = contract.model.model_validate_json(text, strict=True)
    except ValidationError as exc:
        return _raise_violation(exc, contract, text, base, prefix, suffix)
    return _to_document(parsed, contract, text, base, prefix)


def _raise_violation(exc: ValidationError, contract, text, base, prefix, suffix) -> AnalysisDocument:
    errors = exc.errors(include_url=False, include_input=False)
    if any(item["type"] == "json_invalid" for item in errors):
        raise StructuredOutputError(
            "malformed_json",
            f"{prefix}the answer is not valid JSON.{suffix}",
            text,
            {**base, "detail": errors[0]["msg"][:200]},
        ) from exc
    unexpected = [".".join(str(part) for part in item["loc"]) for item in errors if item["type"] == "extra_forbidden"]
    missing = [".".join(str(part) for part in item["loc"]) for item in errors if item["type"] == "missing"]
    code = "unexpected_keys" if unexpected else "missing_fields" if missing else "schema_violation"
    first = errors[0]
    where = ".".join(str(part) for part in first["loc"])
    raise StructuredOutputError(
        code,
        f"{prefix}the answer broke {contract.name}: {len(errors)} errors, first at {where} ({first['msg']}).{suffix}",
        text,
        {
            **base,
            "error_count": len(errors),
            "unexpected_key_count": len(unexpected),
            "unexpected_examples": unexpected[:5],
            "missing_examples": missing[:5],
        },
    ) from exc


def _to_document(parsed: BaseModel, contract, text, base, prefix) -> AnalysisDocument:
    """The one conversion from a model's answer to the stored document."""
    payload = _with_provenance(parsed.model_dump(mode="json"))
    try:
        return AnalysisDocument.model_validate(payload)
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_input=False)
        first = errors[0]
        where = ".".join(str(part) for part in first["loc"])
        raise StructuredOutputError(
            "schema_violation",
            f"{prefix}the answer broke {contract.name}: {len(errors)} values out of range, "
            f"first at {where} ({first['msg']}).",
            text,
            {**base, "error_count": len(errors)},
        ) from exc


def _with_provenance(node: Any) -> Any:
    if isinstance(node, list):
        return [_with_provenance(item) for item in node]
    if not isinstance(node, dict):
        return node
    result = {key: _with_provenance(value) for key, value in node.items()}
    if isinstance(result.get("evidence"), dict):
        result["evidence"] = {**result["evidence"], "provenance": "AI_PREDICTION"}
    return result
