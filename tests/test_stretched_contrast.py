from itertools import product
import logging
from math import floor


import cv2
from docoskin.image_operations import stretched_contrast
import numpy
import pytest


@pytest.mark.parametrize("lower_percentile,upper_percentile,uncovered_rows,shuffle_pixels", tuple(product(
    (0, 1, 5, 14.3, 45,),
    (77, 90, 92.7, 99, 100,),
    (0, 13, 121,),
    (False, True,),
)))
def test_stretched_contrast(lower_percentile, upper_percentile, uncovered_rows, shuffle_pixels):
    image = numpy.broadcast_to(numpy.arange(256, dtype="uint8"), (256, 256))

    if shuffle_pixels:
        # this should make no difference to the assertion phase, so no additional changes needed beyond this
        numpy.random.seed(1234)
        flattened_image = image.flatten()
        numpy.random.shuffle(flattened_image)
        image = flattened_image.reshape(image.shape)

    if uncovered_rows:
        # place `uncovered_rows` black rows at the bottom of the image
        padded_image = numpy.concatenate((image, numpy.zeros((uncovered_rows, 256,), dtype="uint8"),), axis=0)
    else:
        padded_image = image

    adjusted_image = stretched_contrast(
        padded_image,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        coverage=1.0 if not uncovered_rows else (256.0/(256+uncovered_rows)),
    )

    if uncovered_rows:
        # strip the uncovered rows back off for analysis
        adjusted_image = adjusted_image[:256,:]

    assert ((adjusted_image == 0) == (image <= floor((float(lower_percentile)/100)*256))).all()
    # the following comparison value is clipped to 255 (via min()) as we are expected to accept 100.0 percent as a
    # maximum makes the input a closed interval, but we of course are working against half-open interval data.
    assert ((adjusted_image == 255) == (image >= min(floor((float(upper_percentile)/100)*256), 255))).all()

    # the 100's cancel out in this expected_new_median calculation
    expected_new_median =  256 * (50.0-lower_percentile)/(upper_percentile-lower_percentile)
    # we should allow a tolerance for quantization difficulties, but this should vary with the amount of "gain" that's
    # expected to be applied to the image
    tolerance = 1.5 * 100.0/(upper_percentile-lower_percentile)
    assert abs(numpy.percentile(adjusted_image,  50.0, interpolation="lower") - expected_new_median) <= tolerance
