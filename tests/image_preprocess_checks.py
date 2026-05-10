import io
import unittest

from PIL import Image

from services.image_preprocess import crop_subtitle_region


def make_image(width: int = 100, height: int = 80) -> bytes:
    image = Image.new("RGB", (width, height), "white")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class CropSubtitleRegionTests(unittest.TestCase):
    def test_crops_from_normalized_region_rect(self) -> None:
        cropped = crop_subtitle_region(
            make_image(),
            region_rect={
                "x": 10,
                "y": 25,
                "width": 50,
                "height": 25,
            },
        )

        self.assertIsNotNone(cropped)
        self.assertEqual(cropped.size, (50, 20))

    def test_legacy_start_end_percent_still_crop_full_width(self) -> None:
        cropped = crop_subtitle_region(make_image(), start_percent=50, end_percent=100)

        self.assertIsNotNone(cropped)
        self.assertEqual(cropped.size, (100, 40))


if __name__ == "__main__":
    unittest.main()
