from __future__ import annotations

from models.style_profile import StyleProfile
from models.subtitle_event import SubtitleEvent
from providers.base import NO_MATCH_JSON
TEXT_DRY_RUN_NO_MATCH_JSON = '{"m":0,"t":"未识别到字幕"}'


def build_style_keyword_summary(description: str) -> str:
    normalized = " ".join(description.strip().split())
    if not normalized:
        return ""
    return normalized


def build_style_refine_prompt(description: str) -> str:
    compact = build_style_keyword_summary(description)
    return (
        "Convert the subtitle style description into a compact English tag list.\n"
        "Return one line only. No markdown. No explanation.\n"
        "Prefer short visual tags separated by '; '.\n"
        "Examples: red_fill; black_border; 2_lines | white_fill; blue_outline; single_line\n"
        f"Input: {compact}"
    )


def build_style_image_analysis_prompt() -> str:
    return (
        "Describe the subtitle style in the image as a compact English tag list.\n"
        "Return one line only. No markdown. No explanation.\n"
        "Prefer short visual tags separated by '; '.\n"
        "Include color, outline/shadow, and single_line or 2_lines when visible."
    )


def build_style_preview_prompt(style_profiles: list[StyleProfile]) -> str:
    return (
        "Style matching preview.\n"
        "Return exactly one line of JSON.\n"
        "Do not output markdown.\n"
        "Do not output explanation.\n"
        "Do not output analysis.\n"
        f"If none of the locked styles match, return exactly {NO_MATCH_JSON}.\n"
        "If a locked style is visible but the frame also contains confusing overlay text, mixed speaker styles, or any ambiguity that should be checked by a human, return exactly this JSON object: {\"m\":1,\"s\":2,\"l\":1,\"r\":1}.\n"
        "If matched normally, return exactly this JSON object: {\"m\":1,\"s\":2,\"l\":1}.\n"
        "Any output other than the JSON object is invalid.\n"
        "Locked styles:\n"
        f"{_style_lines(style_profiles)}"
    )


def _style_lines(style_profiles: list[StyleProfile]) -> str:
    lines = []
    for profile in style_profiles:
        desc = profile.feature_notes.strip() or "no extra notes"
        lines.append(f"{profile.style_id}: {profile.display_name}; layout={profile.layout_hint}; notes={desc}")
    return "\n".join(lines)


def build_style_job_prompt(
    event: SubtitleEvent,
    style_profiles: list[StyleProfile],
    *,
    sample_image_count: int = 0,
) -> str:
    parts = [
        "You are matching hard subtitle style against a fixed closed set.",
        "Return exactly one line of JSON.",
        "Do not output markdown.",
        "Do not output explanation.",
        "Do not output analysis.",
        "Do not output subtitle text.",
        f"If none of the locked styles match, return exactly {NO_MATCH_JSON}.",
        'If one locked style matches clearly, return exactly this JSON object: {"m":1,"s":2,"l":1}.',
        'If a locked style is present but the frame also includes non-subtitle overlay text, multiple speaker subtitle styles at the same time, or any ambiguity that should be checked by a human, return exactly this JSON object: {"m":1,"s":2,"l":1,"r":1}.',
        "s is the numeric style id. l is 1 or 2 for single-line or double-line subtitle. r is 1 when human review is required.",
        "Any output other than the JSON object is invalid.",
        "Use r=1 conservatively when the frame partially matches a locked style but could be polluted by source-material text or mixed subtitle styles.",
    ]
    if sample_image_count > 0:
        parts.append(
            f"The first {sample_image_count} image(s) are reference examples of locked styles, labeled with their style id. "
            "Use them as visual guides for matching. "
            "The last image is the actual frame to classify."
        )
    parts.append(f"Current subtitle event text from ASS for weak reference only: {event.text}")
    parts.append("Locked styles:")
    parts.append(_style_lines(style_profiles))
    return "\n".join(parts)


def _build_language_instruction(subtitle_language: str) -> str:
    mapping = {
        "auto": "Detect the subtitle language automatically. Do not translate.",
        "zh-Hans": "The subtitle language is Simplified Chinese. Do not translate. Keep Simplified Chinese characters.",
        "zh-Hant": "The subtitle language is Traditional Chinese. Do not translate. Keep Traditional Chinese characters.",
        "ja": "The subtitle language is Japanese. Do not translate. Keep Japanese scripts as shown.",
        "en": "The subtitle language is English. Do not translate. Keep the original capitalization when visible.",
        "ko": "The subtitle language is Korean. Do not translate. Keep Hangul as shown.",
        "mixed": "The subtitle may contain mixed languages such as Chinese, Japanese, English, or Korean. Do not translate. Preserve the visible script of each segment.",
    }
    return mapping.get(subtitle_language, mapping["auto"])


def build_text_job_prompt(event: SubtitleEvent, subtitle_language: str = "auto") -> str:
    return (
        "Read the hard subtitle text from the image.\n"
        "Return JSON only. No markdown. No explanation.\n"
        f"{_build_language_instruction(subtitle_language)}\n"
        "Preserve visible punctuation, spacing, and line breaks when reliable.\n"
        "Do not summarize, rewrite, or normalize the text.\n"
        f"If no reliable subtitle text can be extracted, return exactly {NO_MATCH_JSON}.\n"
        'If matched, return exactly this shape: {"m":1,"t":"subtitle text"}.\n'
        f"Current ASS subtitle text for reference only: {event.text}"
    )


def build_text_dry_run_prompt(subtitle_language: str = "auto") -> str:
    return (
        "Read the hard subtitle text from the image.\n"
        "Return JSON only. No markdown. No explanation.\n"
        f"{_build_language_instruction(subtitle_language)}\n"
        "Preserve visible punctuation, spacing, and line breaks when reliable.\n"
        "Do not summarize, rewrite, or normalize the text.\n"
        'If subtitle text is clearly visible, return exactly this shape: {"m":1,"t":"subtitle text"}.\n'
        f"If there is no subtitle in the selected region, return exactly {TEXT_DRY_RUN_NO_MATCH_JSON}.\n"
        "If only part of the subtitle is visible, infer conservatively and return the most likely full text."
    )
