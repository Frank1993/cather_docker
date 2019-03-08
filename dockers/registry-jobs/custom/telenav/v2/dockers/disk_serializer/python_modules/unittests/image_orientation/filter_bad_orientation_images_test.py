import os
import unittest

from classification.dev.image_orientation.utils.filter_bad_orientation_images import BadOrientationImageFilter
from unittests.common.abstract_base_test import AbstractBaseTest
from unittests.utils import resource_utils as resource_utils


class FilterBadOrientationImagesTest(AbstractBaseTest, unittest.TestCase):
    resources_folder = "correct_orientation"
    ftp_path = "/ORBB/data/test/python/classification/image_orientation/correct_orientation_imgs.zip"

    def test_correct_orientation(self):
        ftp_bundle_path = "/ORBB/data/image_orientation/good_bundle.zip"
        input_folder = os.path.join(resource_utils.LOCAL_TEST_RESOURCES_FOLDER, self.resources_folder ,"orientation_test_images")
        output_folder = os.path.join(resource_utils.LOCAL_TEST_RESOURCES_FOLDER, self.resources_folder ,"orientation_output")

        BadOrientationImageFilter(ftp_bundle_path, input_folder, output_folder).filter_bad_orietation_images()

        nr_moved_files = len(os.listdir(output_folder))

        # move all bad images
        assert (nr_moved_files == 9)