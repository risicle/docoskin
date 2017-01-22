from __future__ import absolute_import

from collections import namedtuple
from itertools import compress
import logging

import cv2
import numpy

from docoskin import defaults
from docoskin.dummythreadpoolexecutor import DummyThreadPoolExecutor
from docoskin.exceptions import DocoskinInvalidArgumentCombinationError, DocoskinNoMatchFoundError


logger = logging.getLogger("docoskin")


FindCandidateHomographyExtendedOutput = namedtuple(
    "FindCandidateHomographyExtendedOutput",
    (
        "reference_keypoints",
        "candidate_keypoints",
        "inlier_matches",
        "outlier_matches",
        "bad_matches",
    ),
)


_KeypointsDescriptorsTuple = namedtuple("KeypointsDescriptorsTuple", ("keypoints", "descriptors"))


def _detect_and_compute(feature_detector_factory, image, name=None):
    r = _KeypointsDescriptorsTuple(*feature_detector_factory().detectAndCompute(image, None))
    logger.debug("Found %i keypoints%s%s", len(r.keypoints), " in " if name else "", name)
    return r


def find_candidate_homography(
        reference_image,
        candidate_image,
        feature_detector_factory=None,
        feature_matcher=None,
        feature_distance_ratio_threshold=defaults.DEFAULT_FEATURE_DISTANCE_RATIO_THRESHOLD,
        n_match_threshold=defaults.DEFAULT_N_MATCH_THRESHOLD,
        reference_keypoints=None,
        reference_descriptors=None,
        ransac_reproj_threshold=defaults.DEFAULT_RANSAC_REPROJ_THRESHOLD,
        ransac_n_inlier_threshold=defaults.DEFAULT_RANSAC_N_INLIER_THRESHOLD,
        thread_pool=None,
        ):
    feature_detector_factory = feature_detector_factory or defaults.default_feature_detector_factory
    feature_matcher = feature_matcher or defaults.default_feature_matcher_factory()
    thread_pool = thread_pool or DummyThreadPoolExecutor()

    candidate_kp_dsc_f = thread_pool.submit(
        _detect_and_compute,
        feature_detector_factory,
        candidate_image,
        "candidate",
    )

    already_trained_descriptors = feature_matcher.getTrainDescriptors()
    if already_trained_descriptors:
        if len(reference_keypoints or ()) != len(already_trained_descriptors):
            raise DocoskinInvalidArgumentCombinationError(
                "Pre-trained feature matchers require a reference_keypoints argument containing the corresponding "
                "keypoints"
            )
    else:
        if not (reference_keypoints or reference_descriptors):
            reference_kp_dsc_f = thread_pool.submit(
                _detect_and_compute,
                feature_detector_factory,
                reference_image,
                "reference",
            )
            # no point in waiting the reference_kp_dsc_f, we need it for the next step
            reference_keypoints, reference_descriptors = reference_kp_dsc_f.result()
        elif reference_keypoints and reference_descriptors:
            if len(reference_keypoints) != len(reference_descriptors):
                raise DocoskinInvalidArgumentCombinationError(
                    "reference_keypoints and reference_descriptors length mismatch"
                )
        else:
            raise DocoskinInvalidArgumentCombinationError(
                "Doesn't make sense to supply reference_keypoints without reference_descriptors or vice-versa"
            )

        feature_matcher.add((reference_descriptors,))
        # so now reference_keypoints and reference_descriptors should have been defined and feature_matcher should have
        # its reference_descriptors one way or another

    matches = feature_matcher.knnMatch(candidate_kp_dsc_f.result().descriptors, k=2)

    # candidate_kp_dsc_f must have returned by now to get to this point, so let's give them some more conventient
    # accessors
    candidate_keypoints, candidate_descriptors = candidate_kp_dsc_f.result()

    good_match_mask = tuple(
        match and (len(match) == 1 or match[0].distance < feature_distance_ratio_threshold*match[1].distance)
        for match in matches
    )
    n_good_matches = sum(1 for good in good_match_mask if good)
    logger.debug("Found %i/%i 'good' keypoint matches", n_good_matches, len(matches))
    if n_good_matches < n_match_threshold:
        raise DocoskinNoMatchFoundError("Not enough 'good' feature matches found ({})".format(n_good_matches))

    reference_coords = numpy.float32(tuple(
        reference_keypoints[match[0].trainIdx].pt for match in compress(matches, good_match_mask)
    )).reshape(-1, 1, 2,)
    candidate_coords = numpy.float32(tuple(
        candidate_keypoints[match[0].queryIdx].pt for match in compress(matches, good_match_mask)
    )).reshape(-1, 1, 2,)

    M, inlier_mask = cv2.findHomography(candidate_coords, reference_coords, cv2.RANSAC, ransac_reproj_threshold)
    n_inliers = sum(inlier_mask.flat)
    if n_inliers < ransac_n_inlier_threshold:
        raise DocoskinNoMatchFoundError("Not enough RANSAC inliers found ({})".format(n_inliers))

    logger.debug("Used %i keypoints as inliers", n_inliers)

    extended_output = FindCandidateHomographyExtendedOutput(
        reference_keypoints=reference_keypoints,
        candidate_keypoints=candidate_keypoints,
        inlier_matches=tuple(
            match[0] for match in compress(compress(matches, good_match_mask), inlier_mask.flat)
        ),
        outlier_matches=tuple(
            match[0] for match in compress(compress(matches, good_match_mask), ((not inlier) for inlier in inlier_mask.flat))
        ),
        bad_matches=tuple(
            match[0] for match in compress(matches, ((not good) for good in good_match_mask))
        ),
    )

    return M, extended_output


def match_and_warp_candidate(reference_image, candidate_image, warp_image=None, **kwargs):
    M = find_candidate_homography(reference_image, candidate_image, **kwargs)[0]
    return cv2.warpPerspective(
        warp_image if warp_image is not None else candidate_image,
        M,
        tuple(reversed(reference_image.shape)),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ), M
