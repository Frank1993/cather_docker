import unittest
from unittests.classification.abstract.abstract_performance_test import AbstractPerformanceTest


class ImageOrientationPerformanceTest(AbstractPerformanceTest, unittest.TestCase):
    ftp_path = "/ORBB/data/test/python/classification/image_orientation/unittest_performance_imgs.zip"
    ftp_bundle_path = "/ORBB/data/image_orientation/good_bundle.zip"
    expected_acc = 1
