import unittest
from unittests.classification.abstract.abstract_performance_test import AbstractPerformanceTest


class RoiClassifierPerformanceTest(AbstractPerformanceTest, unittest.TestCase):
    ftp_path = "/ORBB/data/test/python/classification/roi_classifier/raw_imgs.zip"
    ftp_bundle_path = "/ORBB/data/roi_classifier/good_bundle.zip"
    expected_acc = 0.9
