from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Mapping, Optional

from PIL import Image


@dataclass(slots=True)
class ImageEncodingOptions:
    format_name: str = "JPEG"
    mime_type: str = "image/jpeg"
    quality: int = 75
    max_edge: int = 640


def crop_subtitle_region(
    img_bytes: bytes,
    start_percent: int = 66,
    end_percent: int = 100,
    region_rect: Mapping[str, float] | None = None,
) -> Optional[Image.Image]:
    if not img_bytes:
        return None
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    width, height = image.size
    if region_rect is not None:
        x_start = int(width * float(region_rect.get("x", 0)) / 100)
        y_start = int(height * float(region_rect.get("y", 0)) / 100)
        x_end = int(width * float(region_rect.get("x", 0) + region_rect.get("width", 100)) / 100)
        y_end = int(height * float(region_rect.get("y", 0) + region_rect.get("height", 100)) / 100)
        x_start = max(0, min(width, x_start))
        y_start = max(0, min(height, y_start))
        x_end = max(x_start, min(width, x_end))
        y_end = max(y_start, min(height, y_end))
        return image.crop((x_start, y_start, x_end, y_end))
    y_start = max(0, min(height, int(height * start_percent / 100)))
    y_end = max(y_start, min(height, int(height * end_percent / 100)))
    return image.crop((0, y_start, width, y_end))


def resize_image(image: Image.Image, max_edge: int) -> Image.Image:
    width, height = image.size
    if max(width, height) <= max_edge:
        return image
    ratio = max_edge / max(width, height)
    return image.resize((max(1, int(width * ratio)), max(1, int(height * ratio))), Image.Resampling.LANCZOS)


def encode_image_base64(image: Image.Image, options: ImageEncodingOptions) -> str:
    output = io.BytesIO()
    save_kwargs = {"format": options.format_name}
    if options.format_name.upper() in {"JPEG", "WEBP"}:
        save_kwargs["quality"] = options.quality
    image.save(output, **save_kwargs)
    return base64.standard_b64encode(output.getvalue()).decode("utf-8")


def preprocess_for_llm(
    img_bytes: bytes,
    options: ImageEncodingOptions,
    start_percent: int = 66,
    end_percent: int = 100,
    region_rect: Mapping[str, float] | None = None,
) -> tuple[str, str]:
    image = crop_subtitle_region(img_bytes, start_percent=start_percent, end_percent=end_percent, region_rect=region_rect)
    if image is None:
        raise ValueError("empty image bytes")
    resized = resize_image(image, options.max_edge)
    return options.mime_type, encode_image_base64(resized, options)
