import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from models.job_result import EventJobResult, StyleJobResult, TextJobResult
from storage.job_store import JobStore


class JobStoreTests(unittest.TestCase):
    def test_save_run_records_success_and_failed_tasks(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store = JobStore(Path(temp_dir))
            results = [
                EventJobResult(
                    event_id="event-1",
                    style_result=StyleJobResult(matched=True, style_id=1),
                    text_result=TextJobResult(matched=True, text="hello", review_required=True, review_reasons=["long text"]),
                    final_action="apply_style",
                ),
                EventJobResult(
                    event_id="event-2",
                    style_result=StyleJobResult(matched=True, style_id=2),
                    text_result=TextJobResult(matched=False, raw_response='{"m":1,"t" "bad"}'),
                    final_action="failed",
                    error_messages=["text_job: bad json"],
                    failed_tasks=["text"],
                ),
            ]

            record = store.save_run(
                video_path="video.mp4",
                subtitle_input_path="input.srt",
                ass_output_path="output.ass",
                results=results,
                event_lookup={},
            )
            loaded = store.load(record["job_id"])

        self.assertEqual(loaded["ass_output_path"], "output.ass")
        self.assertEqual(loaded["summary"]["failed_events"], 1)
        self.assertEqual(loaded["events"][0]["tasks"]["style"]["status"], "success")
        self.assertEqual(loaded["events"][0]["tasks"]["text"]["status"], "success")
        self.assertTrue(loaded["events"][0]["tasks"]["text"]["review_required"])
        self.assertEqual(loaded["events"][0]["tasks"]["text"]["review_reasons"], ["long text"])
        self.assertEqual(loaded["events"][1]["tasks"]["style"]["status"], "success")
        self.assertEqual(loaded["events"][1]["tasks"]["text"]["status"], "failed")
        self.assertEqual(loaded["events"][1]["tasks"]["text"]["raw_response"], '{"m":1,"t" "bad"}')

    def test_latest_failed_tasks_map_uses_output_index(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store = JobStore(Path(temp_dir))
            store.save_run(
                video_path="video.mp4",
                subtitle_input_path="input.srt",
                ass_output_path="output.ass",
                results=[
                    EventJobResult(event_id="event-1", failed_tasks=["style"], error_messages=["style failed"]),
                    EventJobResult(event_id="event-2", failed_tasks=["text"], error_messages=["text failed"]),
                ],
                event_lookup={},
            )

            latest = store.load_latest_for_output("output.ass")
            failed_tasks = store.failed_tasks_map(latest)

        self.assertEqual(failed_tasks, {"event-1": ["style"], "event-2": ["text"]})

    def test_retry_record_preserves_previous_success_for_skipped_tasks(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store = JobStore(Path(temp_dir))
            first = store.save_run(
                video_path="video.mp4",
                subtitle_input_path="input.srt",
                ass_output_path="output.ass",
                results=[
                    EventJobResult(
                        event_id="event-1",
                        style_result=StyleJobResult(matched=True, style_id=1),
                        text_result=TextJobResult(matched=True, text="done"),
                    ),
                    EventJobResult(
                        event_id="event-2",
                        style_result=StyleJobResult(matched=True, style_id=2),
                        failed_tasks=["text"],
                        error_messages=["text failed"],
                    ),
                ],
                event_lookup={},
            )
            retry = store.save_run(
                video_path="video.mp4",
                subtitle_input_path="input.srt",
                ass_output_path="output.ass",
                results=[
                    EventJobResult(event_id="event-1"),
                    EventJobResult(event_id="event-2", text_result=TextJobResult(matched=True, text="fixed")),
                ],
                event_lookup={},
                previous_record=first,
            )

        event_1, event_2 = retry["events"]
        self.assertEqual(event_1["tasks"]["style"]["status"], "success")
        self.assertEqual(event_1["tasks"]["text"]["status"], "success")
        self.assertEqual(event_2["tasks"]["style"]["status"], "success")
        self.assertEqual(event_2["tasks"]["text"]["status"], "success")


if __name__ == "__main__":
    unittest.main()
