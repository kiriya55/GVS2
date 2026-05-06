from __future__ import annotations

from providers.base import ProviderCapabilities, ProviderConfig, ProviderResponse, ProviderUsage, estimate_cost


class AnthropicVisionProvider:
    def __init__(self, config: ProviderConfig) -> None:
        import anthropic

        self.config = config
        self.capabilities = ProviderCapabilities(supports_webp=True, preferred_lossy_format="jpeg", supports_prompt_caching=True)
        kwargs = {"api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url.rstrip("/")
        self.client = anthropic.Anthropic(**kwargs)

    def classify(self, prompt: str, images: list[tuple[str, str]]) -> ProviderResponse:
        content = [{"type": "text", "text": prompt}]
        for mime_type, data in images:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": data,
                    },
                }
            )
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_output_tokens,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            messages=[{"role": "user", "content": content}],
            cache_control={"type": "ephemeral"},
        )
        usage_data = getattr(response, "usage", None)
        usage = ProviderUsage(
            input_tokens=int(getattr(usage_data, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage_data, "output_tokens", 0) or 0),
            cache_creation_input_tokens=int(getattr(usage_data, "cache_creation_input_tokens", 0) or 0),
            cache_read_input_tokens=int(getattr(usage_data, "cache_read_input_tokens", 0) or 0),
        )
        usage.estimated_cost_usd = estimate_cost(self.config.model, {}, usage)
        text_blocks = [block.text for block in response.content if block.type == "text"]
        return ProviderResponse(text="\n".join(text_blocks).strip(), usage=usage)
