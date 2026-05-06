from __future__ import annotations

import logging
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
    subtitle_language: str = "auto",
    style_image_options: ImageEncodingOptions | None = None,
    text_image_options: ImageEncodingOptions | None = None,
    progress_callback=None,
    event_ids: set[str] | None = None,
    include_samples: bool = False,
) -> list[EventJobResult]:
    logger.info(f"run_gvs2: 开始处理，视频={video_path}, 输入={ass_input_path}, 输出={ass_output_path}")
    logger.info(f"run_gvs2: 样式配置数={len(style_profiles)}, style_provider={'有' if style_provider else '无'}, text_provider={'有' if text_provider else '无'}")
    
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
        subtitle_language=subtitle_language,
        style_image_options=style_image_options or ImageEncodingOptions(),
        text_image_options=text_image_options or ImageEncodingOptions(max_edge=768),
        style_job=JobSettings(enabled=style_provider is not None, provider_config=style_provider, include_samples=include_samples),
        text_job=JobSettings(enabled=text_provider is not None, provider_config=text_provider),
    )
    pipeline = EventPipeline(settings)
    
    if event_ids is not None:
        logger.info(f"run_gvs2: 指定重跑事件数={len(event_ids)}")
        document.events = [event for event in document.events if event.event_id in event_ids]
        if not document.events:
            raise ValueError("失败事件清单中没有可复跑的 event_id")
        if getattr(document, "event_indices", None):
            document.event_indices = [event.line_index for event in document.events]
        logger.info(f"run_gvs2: 重跑模式，实际处理事件数={len(document.events)}")
    
    logger.info(f"run_gvs2: 开始Pipeline处理，事件数={len(document.events)}")
    results = pipeline.run(video_path, document, style_profiles, progress_callback=progress_callback)
    logger.info(f"run_gvs2: Pipeline处理完成，返回结果数={len(results)}")
    
    AssWriter().write(document, ass_output_path)
    logger.info(f"run_gvs2: 写入输出完成，{ass_output_path}")
    return results
