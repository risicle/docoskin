from __future__ import absolute_import

import cv2


DEFAULT_AKAZE_THRESHOLD=0.005
DEFAULT_FEATURE_DISTANCE_RATIO_THRESHOLD=0.7
DEFAULT_N_MATCH_THRESHOLD=10
DEFAULT_RANSAC_REPROJ_THRESHOLD=5.0
DEFAULT_RANSAC_N_INLIER_THRESHOLD=4
DEFAULT_REMOVED_COLOR_MATRIX=((1.0, 1.0, 0.0,), (0.0, 0.0, 1.0,),)
DEFAULT_ADDED_COLOR_MATRIX=((0.0, 1.0, 0.0,), (1.0, 0.0, 1.0,),)
DEFAULT_CONTRAST_STRETCH_LOWER_PERCENTILE=3.0
DEFAULT_CONTRAST_STRETCH_UPPER_PERCENTILE=92.5


def default_feature_detector_factory():
    return cv2.AKAZE_create(threshold=DEFAULT_AKAZE_THRESHOLD)


def default_bruteforce_feature_matcher_factory():
    return cv2.BFMatcher(normType=cv2.NORM_HAMMING)


def default_flann_feature_matcher_factory():
    return cv2.FlannBasedMatcher(
        indexParams={
            "algorithm": 6,  # FLANN_INDEX_LSH
            "table_number": 6,
            "key_size": 14,
            "multi_probe_level": 1,
        },
        searchParams={
            "checks": 32,
        },
    )


default_feature_matcher_factory = default_flann_feature_matcher_factory
