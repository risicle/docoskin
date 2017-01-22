from __future__ import absolute_import

from itertools import product

import logging

import cv2
import numpy

from docoskin import defaults


logger = logging.getLogger("docoskin")


def coverage_from_candidate_warp(reference_image_shape, candidate_image_shape, M):
    # first check if given this projection M, any corners of reference_image lie outside the extents of candidate_image
    if all(
            0 <= x < candidate_image_shape[1] and 0 <= y < candidate_image_shape[0]
            for x, y in cv2.perspectiveTransform(
                # did i mention perspectiveTransform expects a weird coordinate array format?
                numpy.float32((tuple(product((0, reference_image_shape[1],), (0, reference_image_shape[0],))),)),
                numpy.linalg.inv(M),
            )[0]
            ):
        # they don't - let's not waste our time calculating anything else
        return 1.0

    # "count" the number of pixels that would make it onto the reference_image canvas
    return cv2.warpPerspective(
        numpy.ones(candidate_image_shape),
        M,
        tuple(reversed(reference_image_shape)),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).sum(dtype="float32") / (reference_image_shape[0] * reference_image_shape[1])


def stretched_contrast(
        image,
        lower_percentile=defaults.DEFAULT_CONTRAST_STRETCH_LOWER_PERCENTILE,
        upper_percentile=defaults.DEFAULT_CONTRAST_STRETCH_UPPER_PERCENTILE,
        coverage=1.0,  # the proportion of the image which is covered by relevant pixels (non-relevant pixels are
                       # expected to be value 0)
        ):
    if lower_percentile is None:
        black_point = 0
    else:
        black_point = numpy.percentile(
            image,
            (100 * (1.0-coverage)) + (lower_percentile*coverage),
            interpolation="lower",
        )
    logger.debug("Black point for image at percentile %r = %s", lower_percentile, black_point)

    if upper_percentile is None:
        white_point = 255
    else:
        white_point = numpy.percentile(
            image,
            (100 * (1.0-coverage)) + (upper_percentile*coverage),
            # "lower" is arguably the right choice here, but the reality is that "lower" is the easier behaviour to
            # predict/test against
            interpolation="lower",
        )
    logger.debug("White point for image at percentile %r = %s", upper_percentile, white_point)

    out_image = numpy.clip(
        # careful to convert to float *before* subtraction otherwise we may get wrap-around from any sub-black-point
        # pixels that become negative if the calculation is done in uint8
        (image.astype("float32")-black_point)*(256.0/(white_point-black_point)),
        0,
        255,
    ).astype("uint8")
    return out_image


def diff_overlay_images(
        reference_image,
        candidate_image,
        # a reminder here that opencv uses BGR order for some reason and this reflected in the order these matrices are
        # specified
        removed_color_matrix=defaults.DEFAULT_REMOVED_COLOR_MATRIX,
        added_color_matrix=defaults.DEFAULT_ADDED_COLOR_MATRIX,
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
