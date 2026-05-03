from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from time import sleep
from typing import Optional

import requests

from models.job_result import EventJobResult
from models.style_profile import StyleProfile, load_sample_images
from providers.base import ProviderConfig
from providers.factory import build_provider
from providers.prompt_builder import build_style_job_prompt, build_text_job_prompt
from providers.response_parser import parse_style_result, parse_text_result
from services.frame_sampler import build_sample_times
from services.image_preprocess import ImageEncodingOptions, preprocess_for_llm
from services.media import extract_frame_ffmpeg
from services.subtitle_parser import AssDocument


logger = logging.getLogger(__name__)


REVIEW_STYLE_NAME = "需核查"


@dataclass(slots=True)
class JobSettings:
    enabled: bool
    provider_config: Optional[ProviderConfig] = None
    include_samples: bool = False


@dataclass(slots=True)
class PipelineSettings:
    subtitle_region_start: int = 66
    subtitle_region_end: int = 100
    subtitle_language: str = "auto"
    style_image_options: ImageEncodingOptions = field(default_factory=ImageEncodingOptions)
    text_image_options: ImageEncodingOptions = field(default_factory=lambda: ImageEncodingOptions(max_edge=768))
    style_job: JobSettings = field(default_factory=lambda: JobSettings(False, None))
    text_job: JobSettings = field(default_factory=lambda: JobSettings(False, None))

    def validate(self) -> None:
        if not self.style_job.enabled and not self.text_job.enabled:
            raise ValueError("at least one task must be enabled")
        if self.style_job.enabled and self.style_job.provider_config is None:
            raise ValueError("style task provider config is missing")
        if self.text_job.enabled and self.text_job.provider_config is None:
            raise ValueError("text task provider config is missing")


class EventPipeline:
    def __init__(self, settings: PipelineSettings) -> None:
        logger.info(f"EventPipeline初始化: style_job={settings.style_job.enabled}, text_job={settings.text_job.enabled}")
        self.settings = settings
        settings.validate()
        self.style_provider = build_provider(settings.style_job.provider_config) if settings.style_job.enabled else None
        self.text_provider = build_provider(settings.text_job.provider_config) if settings.text_job.enabled else None
        if self.style_provider:
            logger.info(f"样式识别Provider: {self.style_provider.config.provider_type} / {self.style_provider.config.model}")
        if self.text_provider:
            logger.info(f"文字提取Provider: {self.text_provider.config.provider_type} / {self.text_provider.config.model}")

    def _is_retryable_error(self, exc: Exception) -> bool:
        if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
            return True
        message = str(exc).lower()
        return "timed out" in message or "timeout" in message or "connection aborted" in message or "temporarily unavailable" in message

    def _classify_with_retry(self, provider, prompt: str, images: list[tuple[str, str]]):
        attempts = 2
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return provider.classify(prompt, images)
            except Exception as exc:
                last_error = exc
                if attempt >= attempts or not self._is_retryable_error(exc):
                    raise
                sleep(1.5)
        if last_error is not None:
            raise last_error
        raise RuntimeError("classification failed without error")

    def _build_images(self, video_path: str, event, image_options: ImageEncodingOptions) -> list[tuple[str, str]]:
        images: list[tuple[str, str]] = []
        sample_times = list(build_sample_times(event, 1))
        logger.debug(f"_build_images: {event.event_id}, 采样时间={sample_times}")
        for sample_ms in sample_times:
            frame_bytes = extract_frame_ffmpeg(video_path, sample_ms / 1000)
            if not frame_bytes:
                logger.debug(f"_build_images: {event.event_id}, 采样时间={sample_ms}ms 未提取到帧")
                continue
            images.append(
                preprocess_for_llm(
                    frame_bytes,
                    image_options,
                    start_percent=self.settings.subtitle_region_start,
                    end_percent=self.settings.subtitle_region_end,
                )
            )
        logger.debug(f"_build_images: {event.event_id}, 返回 {len(images)} 张图像")
        return images

    def _process_style_event(self, video_path: str, event, style_profiles: list[StyleProfile], sample_images: list[tuple[str, str]] | None = None) -> EventJobResult:
        logger.debug(f"_process_style_event: 开始处理 {event.event_id}")
        event_result = EventJobResult(event_id=event.event_id)
        style_images = self._build_images(video_path, event, self.settings.style_image_options)
        if not style_images:
            logger.warning(f"_process_style_event: {event.event_id} 无可用图像，跳过")
            event_result.final_action = "skip"
            return event_result

        all_images = list(sample_images or []) + style_images
        prompt = build_style_job_prompt(event, style_profiles, sample_image_count=len(sample_images or []))
        try:
            response = self._classify_with_retry(self.style_provider, prompt, all_images)
            event_result.style_result = parse_style_result(response.text)
            event_result.style_result.usage = response.usage
            if event_result.style_result.matched:
                match = next((profile for profile in style_profiles if profile.style_id == event_result.style_result.style_id), None)
                if match is not None:
                    if event_result.style_result.review_required:
                        event.style = REVIEW_STYLE_NAME
                        event_result.final_action = "review_style"
                        logger.info(f"_process_style_event: {event.event_id} 匹配到样式 '{match.ass_style_name}', 需核查")
                    else:
                        event.style = match.ass_style_name
                        event_result.final_action = "apply_style"
                        logger.info(f"_process_style_event: {event.event_id} 匹配到样式 '{match.ass_style_name}'")
                else:
                    logger.warning(f"_process_style_event: {event.event_id} 匹配到未知style_id={event_result.style_result.style_id}")
                    event_result.final_action = "skip"
            else:
                logger.info(f"_process_style_event: {event.event_id} 未匹配到样式")
                event_result.final_action = "skip"
        except Exception as exc:
            logger.error(f"_process_style_event: {event.event_id} 处理失败: {exc}")
            event_result.error_messages.append(f"style_job: {exc}")
            if event_result.final_action == "skip":
                event_result.final_action = "failed"
        return event_result

    def _process_text_event(self, video_path: str, event) -> EventJobResult:
        logger.debug(f"_process_text_event: 开始处理 {event.event_id}")
        event_result = EventJobResult(event_id=event.event_id)
        text_images = self._build_images(video_path, event, self.settings.text_image_options)
        if not text_images:
            logger.warning(f"_process_text_event: {event.event_id} 无可用图像，跳过")
            event_result.final_action = "skip"
            return event_result
        
        try:
            response = self._classify_with_retry(self.text_provider, build_text_job_prompt(event, self.settings.subtitle_language), text_images)
            event_result.text_result = parse_text_result(response.text)
            event_result.text_result.usage = response.usage
            if event_result.text_result.matched:
                event.set_text(event_result.text_result.text)
                event_result.final_action = "text_only"
                logger.info(f"_process_text_event: {event.event_id} 文字提取成功: '{event_result.text_result.text[:30]}...'")
            else:
                logger.info(f"_process_text_event: {event.event_id} 未匹配到文字")
                event_result.final_action = "skip"
        except Exception as exc:
            logger.error(f"_process_text_event: {event.event_id} 处理失败: {exc}")
            event_result.error_messages.append(f"text_job: {exc}")
            if event_result.final_action == "skip":
                event_result.final_action = "failed"
        return event_result

    def _merge_results(self, base: EventJobResult, extra: EventJobResult) -> EventJobResult:
        if extra.style_result is not None:
            base.style_result = extra.style_result
        if extra.text_result is not None:
            base.text_result = extra.text_result
        if extra.final_action != "skip":
            if extra.final_action == "failed" and base.final_action in {"apply_style", "review_style", "text_only"}:
                pass
            elif extra.final_action == "text_only" and base.final_action in {"apply_style", "review_style"}:
                pass
            else:
                base.final_action = extra.final_action
        base.error_messages.extend(extra.error_messages)
        return base

    def run(self, video_path: str, ass_document: AssDocument, style_profiles: list[StyleProfile], progress_callback=None) -> list[EventJobResult]:
        total = len(ass_document.events)
        logger.info(f"EventPipeline.run: 开始处理，视频={video_path}, 字幕数={total}")
        results_map = {event.event_id: EventJobResult(event_id=event.event_id) for event in ass_document.events}
        completed = 0

        def emit_progress(message: str) -> None:
            if progress_callback is not None:
                progress_callback(completed, total, message)

        if self.style_provider is not None:
            logger.info(f"run: 启动样式识别，共 {total} 条，并发数={max(1, self.style_provider.config.concurrency)}")
            emit_progress(f"开始样式识别，共 {total} 条")
            sample_images: list[tuple[str, str]] = []
            if self.settings.style_job.include_samples:
                raw_samples = load_sample_images(style_profiles)
                for _style_id, mime, data in raw_samples:
                    sample_images.append((mime, data))
                if sample_images:
                    logger.info(f"run: 已加载 {len(sample_images)} 张样式样本图作为 few-shot 参考")
            max_workers = max(1, self.style_provider.config.concurrency)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {executor.submit(self._process_style_event, video_path, event, style_profiles, sample_images): event for event in ass_document.events}
                for future in as_completed(future_map):
                    event_result = future.result()
                    self._merge_results(results_map[event_result.event_id], event_result)
                    completed += 1
                    emit_progress(f"样式识别进度 {completed}/{total}")
            logger.info(f"run: 样式识别完成，已处理 {completed}/{total}")

        completed = 0
        if self.text_provider is not None:
            logger.info(f"run: 启动文字提取，共 {total} 条，并发数={max(1, self.text_provider.config.concurrency)}")
            emit_progress(f"开始文字提取，共 {total} 条")
            max_workers = max(1, self.text_provider.config.concurrency)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {executor.submit(self._process_text_event, video_path, event): event for event in ass_document.events}
                for future in as_completed(future_map):
                    event_result = future.result()
                    self._merge_results(results_map[event_result.event_id], event_result)
                    completed += 1
                    emit_progress(f"文字提取进度 {completed}/{total}")
            logger.info(f"run: 文字提取完成，已处理 {completed}/{total}")

        # 统计最终结果
        action_counts = {}
        error_count = 0
        for result in results_map.values():
            action = result.final_action
            action_counts[action] = action_counts.get(action, 0) + 1
            if result.error_messages:
                error_count += 1
        
        logger.info(f"EventPipeline.run: 处理完成，结果统计: {action_counts}")
        if error_count > 0:
            logger.warning(f"EventPipeline.run: 有 {error_count} 个事件处理出错")
        
        return [results_map[event.event_id] for event in ass_document.events]
