import logging
import os


import cv2
from docoskin import docoskin
import numpy
import pytest


from utils import open_test_image, read_test_image


_words_to_avoid_bboxes = (
    ((132, 2604), (146, 2618),),
    ((159, 2634), (173, 2648),),
    ((159, 2604), (173, 2618),),
    ((132, 2634), (146, 2648),),
)
_why_ten_years_bboxes = (
    ((755, 24), (768, 37),),
    ((782, 54), (795, 67),),
    ((782, 24), (795, 37),),
    ((755, 54), (768, 67),),
)


@pytest.mark.parametrize("reference_path,candidate_path,black_bbox,white_bbox,added_bbox,removed_bbox", (
    (
        "words-to-avoid/reference.png",
        "words-to-avoid/candidate01.png",
    ) + _words_to_avoid_bboxes,
    (
        "words-to-avoid/reference.png",
        "words-to-avoid/candidate02.png",
    ) + _words_to_avoid_bboxes,
    (
        "words-to-avoid/reference.png",
        "words-to-avoid/candidate03.png",
    ) + _words_to_avoid_bboxes,
    (
        "words-to-avoid/reference.png",
        "words-to-avoid/candidate04.png",
    ) + _words_to_avoid_bboxes,
    (
        "words-to-avoid/reference.png",
        "words-to-avoid/candidate05.png",
    ) + _words_to_avoid_bboxes,
    (
        "why-ten-years/reference.png",
        "why-ten-years/candidate01.png",
    ) + _why_ten_years_bboxes,
    (
        "why-ten-years/reference.png",
        "why-ten-years/candidate02.png",
    ) + _why_ten_years_bboxes,
    (
        "why-ten-years/reference.png",
        "why-ten-years/candidate03.png",
    ) + _why_ten_years_bboxes,
))
def test_full_pipeline(tmpdir, reference_path, candidate_path, black_bbox, white_bbox, added_bbox, removed_bbox):
    reference_image = read_test_image(reference_path)
    reference_image_shape = reference_image.shape
    del reference_image  # allow python to recover this memory

    reference_image_file = open_test_image(reference_path)
    candidate_image_file = open_test_image(candidate_path)

    out_file = tmpdir.join("out.png").ensure(file=True).open("r+")

    logging.basicConfig(level=logging.DEBUG)
    docoskin(reference_image_file, candidate_image_file, out_file)

    out_file.seek(0)

    recovered_image = cv2.imdecode(numpy.fromfile(out_file, dtype="uint8"), cv2.IMREAD_COLOR)
    assert recovered_image.shape[:2] == reference_image_shape[:2]

    for ((x0, y0), (x1, y1)), target_bgr in (
            (black_bbox, (0, 0, 0,)),
            (white_bbox, (255, 255, 255,)),
            (added_bbox, (0, 255, 0,)),
            (removed_bbox, (0, 0, 255,)),
        ):
        region_pixels = recovered_image[y0:y1,x0:x1].reshape((-1,3))
        pixel_value_deltas = numpy.float32(target_bgr) - region_pixels
        pixel_value_distances = numpy.sqrt(numpy.sum(pixel_value_deltas**2, axis=1))
        assert numpy.percentile(pixel_value_distances, 75.0, interpolation="lower") < 16
