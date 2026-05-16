from __future__ import annotations

import logging
from pathlib import Path

from models.job_result import EventJobResult
from models.style_profile import StyleProfile
from pipeline.event_pipeline import EventPipeline, JobSettings, PipelineSettings
from providers.base import ProviderConfig
from services.ass_writer import AssWriter
from services.image_preprocess import ImageEncodingOptions
from services.media import probe_video_resolution_ffprobe
from services.subtitle_parser import GeneratedAssDocument, parse_subtitle_document


logger = logging.getLogger(__name__)


def run_gvs2(
    video_path: str,
    ass_input_path: str,
    ass_output_path: str,
    style_profiles: list[StyleProfile],
    style_provider: ProviderConfig | None = None,
    text_provider: ProviderConfig | None = None,
    subtitle_region_start: int = 66,
    subtitle_region_end: int = 100,
    subtitle_region_rect: dict[str, float] | None = None,
    frame_concurrency: int = 5,
    subtitle_language: str = "auto",
    style_image_options: ImageEncodingOptions | None = None,
    text_image_options: ImageEncodingOptions | None = None,
    progress_callback=None,
    event_ids: set[str] | None = None,
    include_samples: bool = False,
    failed_tasks_map: dict[str, list[str]] | None = None,
    output_style_section_lines: list[str] | None = None,
) -> list[EventJobResult]:
    logger.info(f"run_gvs2: 开始处理，视频={video_path}, 输入={ass_input_path}, 输出={ass_output_path}")
    logger.info(f"run_gvs2: 样式配置数={len(style_profiles)}, style_provider={'有' if style_provider else '无'}, text_provider={'有' if text_provider else '无'}")

    input_path = Path(ass_input_path)
    output_path = Path(ass_output_path)
    retry_mode = failed_tasks_map is not None
    retry_from_srt = retry_mode and input_path.suffix.lower() == ".srt"
    using_existing_output_base = False
    if retry_mode and output_path.suffix.lower() == ".ass" and output_path.exists():
        logger.info(f"run_gvs2: 复跑模式，使用已有输出ASS作为基底={ass_output_path}")
        document = parse_subtitle_document(ass_output_path)
        using_existing_output_base = True
    else:
        document = parse_subtitle_document(ass_input_path)
    logger.info(f"run_gvs2: 解析完成，输入字幕数={len(document.events)}")
    
    if isinstance(document, GeneratedAssDocument):
        resolution = probe_video_resolution_ffprobe(video_path) or (1920, 1080)
        document.configure_render_profile(
            resolution[0],
            resolution[1],
            subtitle_region_start=subtitle_region_start,
            subtitle_region_end=subtitle_region_end,
        )
        logger.info(f"run_gvs2: 配置渲染参数，分辨率={resolution}")
    
    settings = PipelineSettings(
        subtitle_region_start=subtitle_region_start,
        subtitle_region_end=subtitle_region_end,
        subtitle_region_rect=subtitle_region_rect or {
            "x": 0.0,
            "y": float(subtitle_region_start),
            "width": 100.0,
            "height": float(max(0, subtitle_region_end - subtitle_region_start)),
        },
        frame_concurrency=frame_concurrency,
        subtitle_language=subtitle_language,
        style_image_options=style_image_options or ImageEncodingOptions(),
        text_image_options=text_image_options or ImageEncodingOptions(max_edge=768),
        style_job=JobSettings(enabled=style_provider is not None, provider_config=style_provider, include_samples=include_samples),
        text_job=JobSettings(enabled=text_provider is not None, provider_config=text_provider),
    )
    pipeline = EventPipeline(settings)
    
    keep_document_whole = (
        failed_tasks_map is not None
        and (isinstance(document, GeneratedAssDocument) or using_existing_output_base or retry_from_srt)
    )
    if event_ids is not None and not keep_document_whole:
        logger.info(f"run_gvs2: 指定重跑事件数={len(event_ids)}")
        document.events = [event for event in document.events if event.event_id in event_ids]
        if not document.events:
            raise ValueError("失败事件清单中没有可复跑的 event_id")
        if getattr(document, "event_indices", None):
            document.event_indices = [event.line_index for event in document.events]
        logger.info(f"run_gvs2: 重跑模式，实际处理事件数={len(document.events)}")
    elif event_ids is not None:
        logger.info(f"run_gvs2: SRT复跑模式，保留完整输出事件数={len(document.events)}，指定复跑事件数={len(event_ids)}")
    
    logger.info(f"run_gvs2: 开始Pipeline处理，事件数={len(document.events)}")
    results = pipeline.run(video_path, document, style_profiles, progress_callback=progress_callback, failed_tasks_map=failed_tasks_map)
    logger.info(f"run_gvs2: Pipeline处理完成，返回结果数={len(results)}")
    
    AssWriter().write(document, ass_output_path, style_section_lines=output_style_section_lines)
    logger.info(f"run_gvs2: 写入输出完成，{ass_output_path}")
    return results
