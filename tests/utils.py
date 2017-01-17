from os.path import join as pathjoin, dirname, abspath


import cv2
import pytest
import py.path


def test_image_abs_path(image_path):
    full_path = py.path.local(abspath(__file__)).dirpath("data/images", image_path)
    if not full_path.check(file=1):
        pytest.skip("Test data file not found. Retrieve docoskin-test-data to run full test suite.")
    return str(full_path)


def read_test_image(image_path):
    return cv2.imread(test_image_abs_path(image_path), cv2.IMREAD_GRAYSCALE)


def open_test_image(image_path):
    return open(test_image_abs_path(image_path), "r")
