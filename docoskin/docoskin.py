from __future__ import absolute_import

from concurrent.futures import ThreadPoolExecutor
import logging

import cv2
import numpy

from docoskin import defaults
from docoskin.image_operations import coverage_from_candidate_warp, diff_overlay_images, stretched_contrast
from docoskin.warping import match_and_warp_candidate


logger = logging.getLogger("docoskin")


_feature_matcher_factories = {
    "bruteforce": defaults.default_bruteforce_feature_matcher_factory,
    "flann": defaults.default_flann_feature_matcher_factory,
}


def docoskin(
        reference_image,
        candidate_image,
        out_file,
        contrast_stretch=True,
        warped_candidate_out_file=None,
        feature_matcher=None,
        thread_pool=None,
        ):
    # we use a combination of numpy and imdecode/imencode for file handling as it allows us to transparently work with
    # any file-like object (including stdin and stdout through "-" options to argparse)
    reference_image = original_reference_image = cv2.imdecode(
        numpy.fromfile(reference_image, dtype="uint8"),
        cv2.IMREAD_GRAYSCALE,
    )
    candidate_image = original_candidate_image = cv2.imdecode(
        numpy.fromfile(candidate_image, dtype="uint8"),
        cv2.IMREAD_GRAYSCALE,
    )

    if contrast_stretch:
        logger.debug("Stretching contrast for reference_image")
        reference_image = stretched_contrast(reference_image)
        logger.debug("Stretching contrast for candidate_image")
        candidate_image = stretched_contrast(candidate_image)

    warped_candidate, M = match_and_warp_candidate(
        reference_image,
        candidate_image,
        warp_image=original_candidate_image,
        feature_matcher=feature_matcher,
        thread_pool=thread_pool,
    )

    if contrast_stretch:
        # the darkest or lightest regions of the candidate image may now have been transformed off the image area so
        # we re-apply the contrast stretching to get results calculated based just on the page area
        coverage = coverage_from_candidate_warp(reference_image.shape, candidate_image.shape, M)
        logger.debug("Warped candidate coverage = %s", coverage)
        logger.debug("Re-stretching contrast for warped candidate")
        warped_candidate = stretched_contrast(warped_candidate, coverage=coverage)
    if warped_candidate_out_file:
        cv2.imencode(".png", warped_candidate)[1].tofile(warped_candidate_out_file)

    overlayed_candidate = diff_overlay_images(reference_image, warped_candidate)

    cv2.imencode(".png", overlayed_candidate)[1].tofile(out_file)


def main():
    import argparse

    def nonneg_int(value):
        n = int(value)
        if n < 0:
            raise ValueError("Value cannot be negative")
        return n

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
        "--matcher", "-m",
        choices=_feature_matcher_factories.keys(),
        default="flann",
        help="Select which feature matcher to use",
    )
    parser.add_argument(
        "--threads", "-t",
        type=nonneg_int,
        default=2,
        metavar="N",
        help="Number of python threads to use (default: %(default)s). 0 disables threading, opencv may use more " \
            "threads internally",
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

    docoskin(
        args.reference_image,
        args.candidate_image,
        args.out_image,
        contrast_stretch=args.contrast_stretch,
        warped_candidate_out_file=args.warped_candidate_out,
        feature_matcher=_feature_matcher_factories[args.matcher](),
        thread_pool=ThreadPoolExecutor(args.threads) if args.threads else None,
    )


if __name__ == "__main__":
    main()
