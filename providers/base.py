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


class VisionProvider(Protocol):
    config: ProviderConfig
    capabilities: ProviderCapabilities

    def classify(self, prompt: str, images: list[tuple[str, str]]) -> ProviderResponse:
        ...
