from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class ProviderConfig:
    provider_type: str
    model: str
    api_key: str
    base_url: str = ""
    timeout_sec: int = 180
    max_output_tokens: int = 256
    concurrency: int = 1
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProviderCapabilities:
    supports_webp: bool = False
    preferred_lossy_format: str = "jpeg"
    supports_prompt_caching: bool = False


@dataclass(slots=True)
class ProviderUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    estimated_cost_usd: float | None = None

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.output_tokens


@dataclass(slots=True)
class ProviderResponse:
    text: str
    usage: ProviderUsage = field(default_factory=ProviderUsage)


NO_MATCH_JSON = '{"m":0}'

_CLAUDE_PRICING_USD_PER_1M: dict[str, tuple[float, float]] = {
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def estimate_cost(model: str, pricing: dict[str, tuple[float, float]], usage: ProviderUsage) -> float | None:
    merged = {**_CLAUDE_PRICING_USD_PER_1M, **pricing}
    prices = merged.get(model)
    if prices is None:
        return None
    input_price, output_price = prices
    return usage.total_input_tokens / 1_000_000 * input_price + usage.output_tokens / 1_000_000 * output_price


class VisionProvider(Protocol):
    config: ProviderConfig
    capabilities: ProviderCapabilities

    def classify(self, prompt: str, images: list[tuple[str, str]]) -> ProviderResponse:
        ...
