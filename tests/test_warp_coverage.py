from itertools import product


import cv2
from docoskin import coverage_from_candidate_warp
import numpy
import pytest


TOLERANCE = 1e-5


@pytest.mark.parametrize("reference_shape,candidate_shape,ref_corners_in_cand,expected_coverage", (
    (
        (200, 100,),
        (400, 300,),
        (
            (10, -10,), (90, -10,),
            (10, 210,), (90, 210),
        ),
        (80.0/100),
    ),
    (
        (2000, 1000,),
        (4000, 3000,),
        (
            (100, -100,), (900, -100,),
            (100, 2100,), (900, 2100),
        ),
        (800.0/1000),
    ),
    (
        (400, 300,),
        (100, 1000,),
        (
            (-10, 10,), (310, 10,),
            (-10, 390,), (310, 390),
        ),
        (380.0/400),
    ),
    (
        (200, 100,),
        (400, 300,),
        (
            (0, 0,), (100, 0,),
            (0, 200,), (100, 200),
        ),
        1.0,
    ),
    (
        (1234, 5678,),
        (2345, 3456,),
        (
            (-123, -234,), (6789, 0,),
            (0, 3456,), (7890, 4567),
        ),
        1.0,
    ),
    (
        (1000, 1000,),
        (550, 800,),
        (
            (500, 0,), (1000, 500,),
            (0, 500,), (500, 1000),
        ),
        0.5,
    ),
    (
        (1000, 1000,),
        (444, 222,),
        (
            (500, -500,), (1500, 500,),
            (-500, 500,), (500, 1500),
        ),
        1.0,
    ),
    (
        (4321, 5432,),
        (6543, 7654,),
        (
            (0, 0,), (-500, -50,),
            (-50, -500,), (-1000, -1000),
        ),
        0.0,
    ),
    (
        (111, 222,),
        (333, 444,),
        (
            (10, 10,), (232, 10,),
            (10, 121,), (232, 121),
        ),
        (101.0*212.0)/(111.0*222.0),
    ),
))
def test_warp_coverage(reference_shape, candidate_shape, ref_corners_in_cand, expected_coverage):
    M = cv2.getPerspectiveTransform(
        # note how we have to do a x, y switch to get product() to iterate in the desired order but return the coords
        # the desired way round
        numpy.float32(tuple((x, y,) for y, x in product((0, candidate_shape[0]),(0, candidate_shape[1])))),
        numpy.float32(ref_corners_in_cand),
    )
    coverage = coverage_from_candidate_warp(reference_shape, candidate_shape, M)
    assert coverage <= 1.0
    assert abs(coverage - expected_coverage) < TOLERANCE
