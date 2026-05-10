import unittest
from concurrent.futures import ThreadPoolExecutor as RealThreadPoolExecutor
from unittest.mock import patch
from types import SimpleNamespace

from models.subtitle_event import SubtitleEvent
from models.style_profile import StyleProfile
from pipeline.event_pipeline import EventPipeline, PipelineSettings
from pipeline.runner import run_gvs2
from providers.base import ProviderConfig
from providers.response_parser import parse_text_result
from services.subtitle_parser import AssDocument, GeneratedAssDocument


def make_event(event_id: str, index: int) -> SubtitleEvent:
    return SubtitleEvent(
        event_id=event_id,
        index=index,
        line_index=index,
        event_type="Dialogue",
        format_fields=["Style", "Text"],
        field_values=["Default", f"text {index}"],
        start_ms=index * 1000,
        end_ms=index * 1000 + 1000,
        text=f"text {index}",
        original_style="Default",
    )


class EventPipelineProgressTests(unittest.TestCase):
    def test_preextract_reports_each_event_progress(self) -> None:
        pipeline = EventPipeline.__new__(EventPipeline)
        pipeline.settings = PipelineSettings()
        provider_config = ProviderConfig(provider_type="openai", model="test", api_key="key", concurrency=1)
        pipeline.style_provider = SimpleNamespace(config=provider_config)
        pipeline.text_provider = SimpleNamespace(config=provider_config)
        pipeline._extract_raw_frames = lambda _video_path, _event: [b"frame"]
        def make_result(event):
            return SimpleNamespace(
                event_id=event.event_id,
                style_result=None,
                text_result=None,
                final_action="skip",
                error_messages=[],
                failed_tasks=[],
            )

        pipeline._process_style_event = lambda _video_path, event, _profiles, _samples, _raw_frames: make_result(event)
        pipeline._process_text_event = lambda _video_path, event, _raw_frames: make_result(event)

        events = [make_event("event-1", 0), make_event("event-2", 1)]
        document = AssDocument(lines=[], event_indices=[], events=events)
        messages: list[str] = []

        pipeline.run("video.mp4", document, [], progress_callback=lambda _current, _total, message: messages.append(message))

        self.assertIn("预提取视频帧 1/2", messages)
        self.assertIn("预提取视频帧 2/2", messages)

    def test_preextract_uses_frame_concurrency_setting(self) -> None:
        captured_workers: list[int] = []

        class CapturingExecutor(RealThreadPoolExecutor):
            def __init__(self, max_workers=None, *args, **kwargs):
                captured_workers.append(max_workers)
                super().__init__(max_workers=max_workers, *args, **kwargs)

        pipeline = EventPipeline.__new__(EventPipeline)
        pipeline.settings = PipelineSettings(frame_concurrency=7)
        provider_config = ProviderConfig(provider_type="openai", model="test", api_key="key", concurrency=1)
        pipeline.style_provider = SimpleNamespace(config=provider_config)
        pipeline.text_provider = SimpleNamespace(config=provider_config)
        pipeline._extract_raw_frames = lambda _video_path, _event: [b"frame"]

        def make_result(event):
            return SimpleNamespace(
                event_id=event.event_id,
                style_result=None,
                text_result=None,
                final_action="skip",
                error_messages=[],
                failed_tasks=[],
            )

        pipeline._process_style_event = lambda _video_path, event, _profiles, _samples, _raw_frames: make_result(event)
        pipeline._process_text_event = lambda _video_path, event, _raw_frames: make_result(event)
        document = AssDocument(lines=[], event_indices=[], events=[make_event("event-1", 0)])

        with patch("pipeline.event_pipeline.ThreadPoolExecutor", CapturingExecutor):
            pipeline.run("video.mp4", document, [])

        self.assertEqual(captured_workers[0], 7)


class RetryRunnerTests(unittest.TestCase):
    def test_srt_retry_keeps_all_events_in_generated_document(self) -> None:
        events = [make_event(f"event-{index + 1}", index) for index in range(3)]
        document = GeneratedAssDocument(events)
        processed_counts: list[int] = []
        written_counts: list[int] = []

        class FakePipeline:
            def __init__(self, _settings) -> None:
                pass

            def run(self, _video_path, ass_document, _style_profiles, progress_callback=None, failed_tasks_map=None):
                processed_counts.append(len(ass_document.events))
                return []

        class FakeWriter:
            def write(self, ass_document, _output_path):
                written_counts.append(len(ass_document.events))
                return _output_path

        with (
            patch("pipeline.runner.parse_subtitle_document", return_value=document),
            patch("pipeline.runner.probe_video_resolution_ffprobe", return_value=(1920, 1080)),
            patch("pipeline.runner.EventPipeline", FakePipeline),
            patch("pipeline.runner.AssWriter", FakeWriter),
        ):
            run_gvs2(
                video_path="video.mp4",
                ass_input_path="input.srt",
                ass_output_path="output.ass",
                style_profiles=[],
                style_provider=None,
                text_provider=ProviderConfig(provider_type="openai", model="test", api_key="key"),
                event_ids={"event-2"},
                failed_tasks_map={"event-2": ["text"]},
            )

        self.assertEqual(processed_counts, [3])
        self.assertEqual(written_counts, [3])


class ResponseParserTests(unittest.TestCase):
    def test_text_result_accepts_literal_newline_inside_json_string(self) -> None:
        result = parse_text_result('{"m":1,"t":"first\nsecond"}')

        self.assertTrue(result.matched)
        self.assertEqual(result.text, "first\\Nsecond")


class StyleFallbackTests(unittest.TestCase):
    def test_style_no_match_applies_default_style(self) -> None:
        pipeline = EventPipeline.__new__(EventPipeline)
        pipeline.settings = PipelineSettings()
        pipeline.style_provider = SimpleNamespace(config=ProviderConfig(provider_type="openai", model="test", api_key="key"), classify=lambda _prompt, _images: SimpleNamespace(text='{"m":0}', usage=None))
        pipeline._build_images = lambda _video_path, _event, _options, _raw_frames=None: [("image/jpeg", "data")]
        event = make_event("event-1", 0)
        event.style = "Fancy"

        result = pipeline._process_style_event("video.mp4", event, [])

        self.assertEqual(event.style, "Default")
        self.assertEqual(result.final_action, "default_style")

    def test_style_review_keeps_suspected_style_and_marks_text(self) -> None:
        pipeline = EventPipeline.__new__(EventPipeline)
        pipeline.settings = PipelineSettings()
        pipeline.style_provider = SimpleNamespace(config=ProviderConfig(provider_type="openai", model="test", api_key="key"), classify=lambda _prompt, _images: SimpleNamespace(text='{"m":1,"s":7,"l":1,"r":1}', usage=None))
        pipeline._build_images = lambda _video_path, _event, _options, _raw_frames=None: [("image/jpeg", "data")]
        event = make_event("event-1", 0)
        profile = StyleProfile(style_id=7, display_name="Suspect", ass_style_name="Suspect", feature_notes="", layout_hint="either")

        result = pipeline._process_style_event("video.mp4", event, [profile])

        self.assertEqual(event.style, "Suspect")
        self.assertTrue(event.text.startswith(r"{\i1\c&H00FFFF&"))
        event.set_text("recognized text")
        self.assertEqual(event.text, r"{\i1\c&H00FFFF&\3c&H000000&}recognized text")
        self.assertEqual(result.final_action, "review_style")


if __name__ == "__main__":
    unittest.main()
