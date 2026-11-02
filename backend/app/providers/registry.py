"""Provider registry. The submission build runs Meta models only.

Fixture, deterministic and manual_json are local paths with no model behind them.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.providers.local import DeterministicProvider, FixtureProvider
from app.providers.meta_provider import MetaProvider
from app.providers.types import ProviderError

PROVIDERS = {
    MetaProvider.id: MetaProvider,
    DeterministicProvider.id: DeterministicProvider,
    FixtureProvider.id: FixtureProvider,
}

_MANUAL_JSON = "manual_json"
_LOCAL = {DeterministicProvider.id, FixtureProvider.id, _MANUAL_JSON}


def get_provider(provider_id: str, session: Session | None = None):
    cls = PROVIDERS.get(provider_id)
    if cls is None:
        raise ProviderError(f"Provider {provider_id} is not part of this build. LearnLoop runs Meta models only.")
    return cls()


def provider_ids() -> list[str]:
    return [*PROVIDERS.keys(), _MANUAL_JSON]


def is_local(provider_id: str) -> bool:
    return provider_id in _LOCAL


def run_transport(provider_id: str, config: dict | None) -> str:
    return "api"


def needs_secret(provider_id: str, transport: str = "api") -> bool:
    return provider_id not in _LOCAL
