from __future__ import annotations

import requests

from providers.base import ProviderCapabilities, ProviderConfig, ProviderResponse, ProviderUsage


_OPENAI_PRICING_USD_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.5, 10.0),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4.1-nano": (0.1, 0.4),
    "gpt-4o-mini": (0.15, 0.6),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


class OpenAICompatibleVisionProvider:
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.capabilities = ProviderCapabilities(supports_webp=True, preferred_lossy_format="jpeg", supports_prompt_caching=False)

    def _estimate_cost(self, usage: ProviderUsage) -> float | None:
        pricing = _OPENAI_PRICING_USD_PER_1M.get(self.config.model)
        if pricing is None:
            return None
        input_price, output_price = pricing
        return usage.total_input_tokens / 1_000_000 * input_price + usage.output_tokens / 1_000_000 * output_price

    def classify(self, prompt: str, images: list[tuple[str, str]]) -> ProviderResponse:
        base_url = self.config.base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            url = base_url
        elif base_url.endswith("/v1") or base_url.endswith("/api/v3"):
            url = base_url + "/chat/completions"
        else:
            url = base_url + "/v1/chat/completions"
        content = [{"type": "text", "text": prompt}]
        for mime_type, data in images:
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{data}"}})
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.config.max_output_tokens,
            "temperature": 0.2,
        }
        if self.config.extra_params.get("disable_thinking"):
            payload["thinking"] = {"type": "disabled"}
        response = requests.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
            timeout=self.config.timeout_sec,
        )
        response.raise_for_status()
        data = response.json()
        usage_data = data.get("usage") or {}
        usage = ProviderUsage(
            input_tokens=int(usage_data.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage_data.get("completion_tokens", 0) or 0),
        )
        usage.estimated_cost_usd = self._estimate_cost(usage)
        text = data["choices"][0]["message"]["content"].strip()
        return ProviderResponse(text=text, usage=usage)
