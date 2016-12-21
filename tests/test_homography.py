from itertools import izip, product
import logging
from math import hypot


import cv2
from docoskin import find_candidate_homography
import numpy
import pytest


from utils import read_test_image


@pytest.mark.parametrize("reference_path,candidate_path,ref_corners_in_cand,tolerance", (
    ("words-to-avoid/reference.png", "words-to-avoid/candidate01.png", (
        (1557, 180), (1426, 895),
        (408, 41), (370, 751),
    ), 5,), # didn't keep the original of this image - hard to determine the "ground truth" corner coords. up tolerance.
    ("words-to-avoid/reference.png", "words-to-avoid/candidate02.png", (
        (2004, 2044), (811, 2044),
        (2004, 484), (811, 484),
    ), 5,),
    ("words-to-avoid/reference.png", "words-to-avoid/candidate03.png", (
        (1793, 2573), (6, 2523),
        (1820, 26), (45, 15),
    ), 5,),
    ("words-to-avoid/reference.png", "words-to-avoid/candidate04.png", (
        (254, 490), (1672, 457),
        (319, 2963), (1925, 2852),
    ), 5,),
    ("words-to-avoid/reference.png", "words-to-avoid/candidate05.png", (
        (2832, 22), (3140, 1368),
        (469, 604), (904, 1930),
    ), 5,),
    ("why-ten-years/reference.png", "why-ten-years/candidate01.png", (
        (-25, -25), (790, 14),
        (13, 1228), (800, 1242),
    ), 5,),
    ("why-ten-years/reference.png", "why-ten-years/candidate02.png", (
        (1301, 998), (627, 991),
        (1242, 11), (605, 27),
    ), 5,),
    ("why-ten-years/reference.png", "why-ten-years/candidate03.png", (
        (629, 285), (1522, 545),
        (272, 1771), (1298, 1970),
    ), 5,),
))
def test_matrix_generation(reference_path, candidate_path, ref_corners_in_cand, tolerance,):
    reference_image = read_test_image(reference_path)
    candidate_image = read_test_image(candidate_path)

    logging.basicConfig(level=logging.DEBUG)
    M = find_candidate_homography(reference_image, candidate_image)[0]

    # opencv wants its coords in a slightly odd format for this call
    ref_corners_in_cand_a = numpy.float32((ref_corners_in_cand,))

    # ref_corners_in_cand is supposed to denote the 4 corners of the reference document in the candidate image
    # specified in the order top left, top right, bottom left, bottom right. so if all is well those coordinates passed
    # through our recovered perspective matrix should be fairly close to reference_image's dimensions
    transformed_coords = cv2.perspectiveTransform(ref_corners_in_cand_a, M)[0]
    assert all(
        # note ref_y, ref_x order switcharoo here because product() would otherwise traverse the corners in the opposite
        # order we want
        hypot(tr_x-ref_x, tr_y-ref_y) < tolerance for (tr_x, tr_y), (ref_y, ref_x) in izip(
            transformed_coords,
            product((0, reference_image.shape[0],), (0, reference_image.shape[1],)),
        )
    )
