import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from services.ass_writer import remove_review_style_from_ass, replace_ass_styles_section
from services.subtitle_parser import GeneratedAssDocument, extract_ass_style_section
from tests.event_pipeline_checks import make_event


class AssStyleSectionTests(unittest.TestCase):
    def test_generated_ass_does_not_include_review_style(self) -> None:
        document = GeneratedAssDocument([make_event("event-1", 0)])
        content = document.dump()

        self.assertNotIn("Style: 需核查,", content)

    def test_replace_ass_styles_section_uses_imported_section_and_removes_review_style(self) -> None:
        content = "\n".join(
            [
                "[Script Info]",
                "ScriptType: v4.00+",
                "",
                "[V4+ Styles]",
                "Format: Name, Fontname, Fontsize",
                "Style: Default,Arial,20",
                "Style: 需核查,Arial,20",
                "",
                "[Events]",
                "Format: Layer, Start, End, Style, Text",
            ]
        )
        imported = [
            "[V4+ Styles]\n",
            "Format: Name, Fontname, Fontsize\n",
            "Style: Imported,Source Han Sans,48\n",
            "Style: 需核查,Arial,20\n",
            "\n",
        ]

        result = replace_ass_styles_section(content, imported)

        self.assertIn("Style: Imported,Source Han Sans,48", result)
        self.assertNotIn("Style: Default,Arial,20", result)
        self.assertNotIn("Style: 需核查,", result)

    def test_remove_review_style_from_existing_ass(self) -> None:
        content = "[V4+ Styles]\nFormat: Name, Fontname\nStyle: 需核查,Arial\nStyle: Keep,Arial\n[Events]\n"

        result = remove_review_style_from_ass(content)

        self.assertNotIn("Style: 需核查,", result)
        self.assertIn("Style: Keep,Arial", result)

    def test_extract_ass_style_section_preserves_original_lines(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "source.ass"
            path.write_text(
                "[Script Info]\n"
                "\n"
                "[V4+ Styles]\n"
                "Format: Name, Fontname\n"
                "Style: Imported,Arial\n"
                "\n"
                "[Events]\n",
                encoding="utf-8",
            )

            section = extract_ass_style_section(str(path))

        self.assertEqual(section[0], "[V4+ Styles]\n")
        self.assertIn("Style: Imported,Arial\n", section)
        self.assertNotIn("[Events]\n", section)


if __name__ == "__main__":
    unittest.main()
