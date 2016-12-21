from os.path import join as pathjoin, dirname, abspath


import cv2


def read_test_image(image_path):
    return cv2.imread(pathjoin(dirname(abspath(__file__)), "data/images", image_path), cv2.IMREAD_GRAYSCALE)


def open_test_image(image_path):
    return open(pathjoin(dirname(abspath(__file__)), "data/images", image_path), "r")
