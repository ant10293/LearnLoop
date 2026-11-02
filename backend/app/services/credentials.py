"""Server-side credential store. Plaintext keys are never returned to clients."""

from __future__ import annotations

import os
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.runs import ProviderSecret, ProviderStatus
from app.models.util import utcnow

ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "meta": "MODEL_API_KEY",
    "openai_compatible": "OPENAI_COMPATIBLE_API_KEY",
    "cursor": "CURSOR_API_KEY",
}

ENV_FALLBACKS = {
    "gemini": ["GOOGLE_API_KEY"],
    "meta": ["META_MODEL_API_KEY"],
}


def _key_path() -> Path:
    path = settings.resolved_data_dir / ".credential_key"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(Fernet.generate_key())
        try:
            path.chmod(0o600)
        except OSError:
            pass
    return path


def _fernet() -> Fernet:
    return Fernet(_key_path().read_bytes())


def env_secret(provider_id: str) -> str:
    primary = ENV_KEYS.get(provider_id, "")
    if primary and os.environ.get(primary):
        return os.environ[primary].strip()
    for name in ENV_FALLBACKS.get(provider_id, []):
        if os.environ.get(name):
            return os.environ[name].strip()
    return ""


def store_secret(session: Session, provider_id: str, secret: str) -> None:
    token = _fernet().encrypt(secret.encode()).decode()
    hint = secret[-4:] if len(secret) >= 4 else ""
    row = session.get(ProviderSecret, provider_id)
    if row is None:
        session.add(ProviderSecret(provider_id=provider_id, ciphertext=token, hint=hint))
    else:
        row.ciphertext = token
        row.hint = hint
        row.updated_at = utcnow()
    session.commit()


def delete_secret(session: Session, provider_id: str) -> None:
    row = session.get(ProviderSecret, provider_id)
    if row is not None:
        session.delete(row)
        session.commit()


def load_secret(session: Session, provider_id: str) -> tuple[str, str]:
    """Return (secret, source) where source is env, store, or none."""
    from_env = env_secret(provider_id)
    if from_env:
        return from_env, "env"
    row = session.get(ProviderSecret, provider_id)
    if row is None:
        return "", "none"
    try:
        return _fernet().decrypt(row.ciphertext.encode()).decode(), "store"
    except InvalidToken:
        return "", "none"


def secret_hint(session: Session, provider_id: str) -> tuple[bool, str, str]:
    env_value = env_secret(provider_id)
    if env_value:
        return True, "env", env_value[-4:]
    row = session.get(ProviderSecret, provider_id)
    if row is None:
        return False, "none", ""
    return True, "store", row.hint


def record_validation(session: Session, provider_id: str, ok: bool, error: str) -> None:
    row = session.get(ProviderStatus, provider_id)
    if row is None:
        row = ProviderStatus(provider_id=provider_id)
        session.add(row)
    row.last_validated_at = utcnow()
    row.last_ok = ok
    row.last_error = "" if ok else error[:2000]
    session.commit()


def provider_status(session: Session, provider_id: str) -> ProviderStatus | None:
    return session.scalar(select(ProviderStatus).where(ProviderStatus.provider_id == provider_id))
