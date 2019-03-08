import unittest

from unittests.classification.abstract.abstract_performance_test import AbstractPerformanceTest

class ImageQualityPerformanceTest(AbstractPerformanceTest, unittest.TestCase):
    ftp_path = "/ORBB/data/test/python/classification/image_quality/unittests_imgs.zip"
    ftp_bundle_path = "/ORBB/data/image_quality/good_bundle.zip"
    expected_acc = 0.962
