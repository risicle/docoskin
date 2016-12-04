from collections import namedtuple
from itertools import compress
import logging

import cv2
import numpy


DEFAULT_AKAZE_THRESHOLD=0.002
DEFAULT_FEATURE_DISTANCE_RATIO_THRESHOLD=0.7
DEFAULT_N_MATCH_THRESHOLD=10
DEFAULT_RANSAC_REPROJ_THRESHOLD=5.0
DEFAULT_RANSAC_N_INLIER_THRESHOLD=4
DEFAULT_REMOVED_COLOR_MATRIX=((1.0, 1.0, 0.0,), (0.0, 0.0, 1.0,),)
DEFAULT_ADDED_COLOR_MATRIX=((0.0, 1.0, 0.0,), (1.0, 0.0, 1.0,),)
DEFAULT_CONTRAST_STRETCH_LOWER_PERCENTILE=2.0
DEFAULT_CONTRAST_STRETCH_UPPER_PERCENTILE=95.0


logger = logging.getLogger("docoskin")


class DocoskinInvalidArgumentCombinationError(TypeError): pass
class DocoskinNoMatchFoundError(Exception): pass


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


def default_feature_detector():
    return cv2.AKAZE_create(threshold=DEFAULT_AKAZE_THRESHOLD)


def default_feature_matcher():
    return cv2.BFMatcher(normType=cv2.NORM_HAMMING)


def find_candidate_homography(
        reference_image,
        candidate_image,
        feature_detector=None,
        feature_matcher=None,
        feature_distance_ratio_threshold=DEFAULT_FEATURE_DISTANCE_RATIO_THRESHOLD,
        n_match_threshold=DEFAULT_N_MATCH_THRESHOLD,
        reference_keypoints=None,
        reference_descriptors=None,
        ransac_reproj_threshold=DEFAULT_RANSAC_REPROJ_THRESHOLD,
        ransac_n_inlier_threshold=DEFAULT_RANSAC_N_INLIER_THRESHOLD,
        ):
    feature_detector = feature_detector or default_feature_detector()
    feature_matcher = feature_matcher or default_feature_matcher()

    candidate_keypoints, candidate_descriptors = feature_detector.detectAndCompute(candidate_image, None)

    logger.debug("Found %i keypoints in candidate", len(candidate_keypoints))

    already_trained_descriptors = feature_matcher.getTrainDescriptors()
    if already_trained_descriptors:
        if len(reference_keypoints or ()) != len(already_trained_descriptors):
            raise DocoskinInvalidArgumentCombinationError(
                "Pre-trained feature matchers require a reference_keypoints argument containing the corresponding "
                "keypoints"
            )
        matches = feature_matcher.knnMatch(candidate_descriptors, k=2)
    else:
        if not (reference_keypoints or reference_descriptors):
            reference_keypoints, reference_descriptors = feature_detector.detectAndCompute(reference_image, None)
            logger.debug("Found %i keypoints in reference", len(reference_keypoints))
        elif reference_keypoints and reference_descriptors:
            if len(reference_keypoints) != len(reference_descriptors):
                raise DocoskinInvalidArgumentCombinationError(
                    "reference_keypoints and reference_descriptors length mismatch"
                )
        else:
            raise DocoskinInvalidArgumentCombinationError(
                "Doesn't make sense to supply reference_keypoints without reference_descriptors or vice-versa"
            )

        # so now reference_keypoints and reference_descriptors should have been defined one way or another
        matches = feature_matcher.knnMatch(candidate_descriptors, reference_descriptors, k=2)

    good_match_mask = tuple(m.distance < feature_distance_ratio_threshold*n.distance for m, n in matches)
    n_good_matches = sum(1 for good in good_match_mask if good)
    if n_good_matches < n_match_threshold:
        raise DocoskinNoMatchFoundError("Not enough 'good' feature matches found ({})".format(n_good_matches))

    logger.debug("Found %i/%i 'good' keypoint matches", n_good_matches, len(matches))

    reference_coords = numpy.float32(tuple(
        reference_keypoints[m.trainIdx].pt for m, n in compress(matches, good_match_mask)
    )).reshape(-1, 1, 2,)
    candidate_coords = numpy.float32(tuple(
        candidate_keypoints[m.queryIdx].pt for m, n in compress(matches, good_match_mask)
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
            m for m, n in compress(compress(matches, good_match_mask), inlier_mask.flat)
        ),
        outlier_matches=tuple(
            m for m, n in compress(compress(matches, good_match_mask), ((not inlier) for inlier in inlier_mask.flat))
        ),
        bad_matches=tuple(
            m for m, n in compress(matches, ((not good) for good in good_match_mask))
        ),
    )

    return M, extended_output


def match_and_warp_candidate(reference_image, candidate_image, **kwargs):
    M = find_candidate_homography(reference_image, candidate_image, **kwargs)[0]
    return cv2.warpPerspective(candidate_image, M, tuple(reversed(reference_image.shape)), flags=cv2.INTER_CUBIC)


def stretched_contrast(
        image,
        lower_percentile=DEFAULT_CONTRAST_STRETCH_LOWER_PERCENTILE,
        upper_percentile=DEFAULT_CONTRAST_STRETCH_UPPER_PERCENTILE,
    ):
    if lower_percentile is None:
        black_point = 0
    elif lower_percentile == 0:
        black_point = numpy.amin(image)
    else:
        black_point = numpy.percentile(image, lower_percentile, interpolation="lower")
    logger.debug("Black point for image at percentile %r = %s", lower_percentile, black_point)

    if upper_percentile is None:
        white_point = 255
    elif upper_percentile == 0:
        white_point = numpy.amax(image)
    else:
        white_point = numpy.percentile(image, upper_percentile, interpolation="higher")
    logger.debug("White point for image at percentile %r = %s", upper_percentile, white_point)

    out_image = numpy.clip(
        # careful to convert to float *before* subtraction otherwise we may get wrap-around from any sub-black-point
        # pixels that become negative if the calculation is done in uint8
        (image.astype("float32")-black_point)*(255.0/(white_point-black_point)),
        0,
        255,
    ).astype("uint8")
    return out_image


def diff_overlay_images(
        reference_image,
        candidate_image,
        # a reminder here that opencv uses BGR order for some reason and this reflected in the order these matrices are
        # specified
        removed_color_matrix=DEFAULT_REMOVED_COLOR_MATRIX,
        added_color_matrix=DEFAULT_ADDED_COLOR_MATRIX,
        ):
    # combine images into a single two-channel image
    stacked_image = numpy.stack((reference_image, candidate_image,), axis=-1)

    # now we can apply a matrix multiplication to produce two RGB images, highlighting the removed and added sections
    # respectively. we are essentially performing a linear transformation of the planar, 2d colorspace into the 3d
    # RGB cube if it help to visualize it that way.
    removed_image = numpy.dot(stacked_image, numpy.float32(removed_color_matrix)).astype("uint8")
    added_image = numpy.dot(stacked_image, numpy.float32(added_color_matrix)).astype("uint8")

    # the final image is produced by selecting between these two RGB images depending on whether each pixel underwent a
    # removal (less black) or an addition (more black). we determine this using a simple pixel value comparison
    return numpy.where(numpy.expand_dims(stacked_image[:,:,0] > stacked_image[:,:,1], -1), added_image, removed_image)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Onion-skin two document images and output resulting png file to stdout"
    )
    parser.add_argument("reference_image", type=argparse.FileType("r"), help="Reference document image file")
    parser.add_argument("candidate_image", type=argparse.FileType("r"), help="Candidate document image file")
    parser.add_argument(
        "out_image",
        type=argparse.FileType("w"),
        nargs="?",
        default="-",
        help="Candidate document image file (default: %(default)s)",
    )
    parser.add_argument(
        "--warped-candidate-out", "-w",
        type=argparse.FileType("w"),
        metavar="FILE",
        help="Output warped (but unmerged) candidate image to FILE",
    )
    parser.add_argument(
        "--no-contrast-stretch", "-C",
        dest="contrast_stretch",
        action="store_false",
        help="Don't perform any contrast-stretching steps on input files",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Emit debugging information")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # we use a combination of numpy and imdecode/imencode for file handling as it allows us to transparently work with
    # stdin and stdout through "-" options to argparse
    reference_image = cv2.imdecode(numpy.fromfile(args.reference_image, dtype="uint8"), cv2.IMREAD_GRAYSCALE)
    candidate_image = cv2.imdecode(numpy.fromfile(args.candidate_image, dtype="uint8"), cv2.IMREAD_GRAYSCALE)

    if args.contrast_stretch:
        logger.debug("Stretching contrast for reference_image")
        reference_image = stretched_contrast(reference_image)
        logger.debug("Stretching contrast for candidate_image")
        candidate_image = stretched_contrast(candidate_image)

    warped_candidate = match_and_warp_candidate(reference_image, candidate_image)
    if args.contrast_stretch:
        # the darkest or lightest regions of the candidate image may now have been transformed off the image area so
        # we re-apply the contrast stretching to get results calculated based just on the page area
        logger.debug("Re-stretching contrast for warped candidate")
        warped_candidate = stretched_contrast(warped_candidate)
    if args.warped_candidate_out:
        cv2.imencode(".png", warped_candidate)[1].tofile(args.warped_candidate_out)

    overlayed_candidate = diff_overlay_images(reference_image, warped_candidate)

    cv2.imencode(".png", overlayed_candidate)[1].tofile(args.out_image)