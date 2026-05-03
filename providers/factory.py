from __future__ import annotations

from providers.anthropic_provider import AnthropicVisionProvider
from providers.base import ProviderConfig, VisionProvider
from providers.openai_provider import OpenAICompatibleVisionProvider


def build_provider(config: ProviderConfig) -> VisionProvider:
    if config.provider_type == "anthropic":
        return AnthropicVisionProvider(config)
    if config.provider_type == "openai":
        return OpenAICompatibleVisionProvider(config)
    raise ValueError(f"unsupported provider type: {config.provider_type}")
