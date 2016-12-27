from itertools import product
import logging
from math import floor


import cv2
from docoskin import stretched_contrast
import numpy
import pytest


@pytest.mark.parametrize("lower_percentile,upper_percentile,shuffle_pixels", tuple(product(
    (5, 15, 45,),
    (77, 90, 92,),
    (False, True,),
)))
def test_stretched_contrast(lower_percentile, upper_percentile, shuffle_pixels):
    logging.basicConfig(level=logging.DEBUG)
    image = numpy.broadcast_to(numpy.arange(256, dtype="uint8"), (256, 256))

    if shuffle_pixels:
        # this should make no difference to the assertion phase, so no additional changes needed beyond this
        numpy.random.seed(1234)
        flattened_image = image.flatten()
        numpy.random.shuffle(flattened_image)
        image = flattened_image.reshape(image.shape)

    adjusted_image = stretched_contrast(image, lower_percentile=lower_percentile, upper_percentile=upper_percentile)

    assert ((adjusted_image == 0) == (image <= floor((float(lower_percentile)/100)*256))).all()
    assert ((adjusted_image == 255) == (image >= floor((float(upper_percentile)/100)*256))).all()
