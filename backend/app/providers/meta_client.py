"""A small httpx client for the Meta Model API (https://dev.meta.ai).

It covers the two endpoints LearnLoop calls: Responses (plain and streamed) and Files.
Nothing retries. A retry is another billable request.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import IO, Any

import httpx

META_BASE = "https://api.meta.ai/v1"


class MetaAPIError(Exception):
    """A non-2xx answer from the Meta Model API."""

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"Meta API {status_code}: {body[:800]}")


class MetaObject(SimpleNamespace):
    """Attribute access over a JSON object, keeping the original dict for model_dump()."""

    def __init__(self, raw: dict):
        super().__init__(**{key: _wrap(value) for key, value in raw.items()})
        object.__setattr__(self, "_raw", raw)

    def model_dump(self) -> dict:
        return dict(self._raw)


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return MetaObject(value)
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    return value


def _output_text(payload: dict) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    parts = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "output_text":
                parts.append(str(part.get("text") or ""))
    return "".join(parts)


async def _raise_for_status(response: httpx.Response) -> None:
    if response.is_success:
        return
    body = (await response.aread()).decode("utf-8", errors="replace")
    raise MetaAPIError(response.status_code, body)


class _Responses:
    def __init__(self, http: httpx.AsyncClient):
        self._http = http

    async def create(self, **body: Any) -> MetaObject | AsyncIterator[dict]:
        if body.get("stream"):
            return self._stream(body)
        response = await self._http.post("/responses", json=body)
        await _raise_for_status(response)
        payload = response.json()
        result = MetaObject(payload)
        result.output_text = _output_text(payload)
        return result

    async def _stream(self, body: dict) -> AsyncIterator[dict]:
        """Server-sent events, one JSON object per data line."""
        async with self._http.stream("POST", "/responses", json=body) as response:
            await _raise_for_status(response)
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                yield json.loads(data)


class _Files:
    def __init__(self, http: httpx.AsyncClient):
        self._http = http

    async def create(self, *, file: IO[bytes], purpose: str) -> MetaObject:
        name = Path(getattr(file, "name", "") or "upload").name
        response = await self._http.post("/files", data={"purpose": purpose}, files={"file": (name, file)})
        await _raise_for_status(response)
        return MetaObject(response.json())


class MetaClient:
    def __init__(
        self,
        api_key: str,
        *,
        timeout: float,
        base_url: str = META_BASE,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,
        )
        self.responses = _Responses(self._http)
        self.files = _Files(self._http)

    async def __aenter__(self) -> MetaClient:
        return self

    async def __aexit__(self, *_exc) -> bool:
        await self._http.aclose()
        return False


def client_version() -> str:
    return f"learnloop-meta-client (httpx {httpx.__version__})"
