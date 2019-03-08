import os
import sys
import unittest

from collections import namedtuple

import apollo_python_common.log_util as log_util
import apollo_python_common.proto_api as proto_api
from math import radians, cos, sin, asin, sqrt
from unittests.common.abstract_base_test import AbstractBaseTest
from unittests.utils import resource_utils as resource_utils
from sign_positioning.utils.first_local_runner import FirstLocalRunner
from apollo_python_common.map_geometry.geometry_utils import compute_haversine_distance

Match = namedtuple(
    'Match', ['expected_roi', 'actual_roi'])


def select_matched_rois(expected_rois, actual_rois):
    matches = list()
    for expected_roi in expected_rois:
        for actual_roi in actual_rois:
            if expected_roi.id == actual_roi.id:
                matches.append(Match(expected_roi, actual_roi))
    return matches


def sign_positioning_distance(expected_roi, actual_roi):
    lat1, lon1 = expected_roi.local.position.latitude, expected_roi.local.position.longitude
    lat2, lon2 = actual_roi.local.position.latitude, actual_roi.local.position.longitude
    return compute_haversine_distance(lon1, lat1, lon2, lat2)


def get_positioning_average_error_meters(expected_dictionary, actual_dictionary):
    sign_distance_error_sum = 0
    sign_distance_count = 0
    for expected_file in expected_dictionary.keys():
        actual_rois = actual_dictionary[expected_file]
        expected_rois = expected_dictionary[expected_file]
        matches = select_matched_rois(expected_rois, actual_rois)
        for match in matches:
            sign_distance = sign_positioning_distance(match.expected_roi, match.actual_roi)
            sign_distance_error_sum += sign_distance
            sign_distance_count += 1

    return sign_distance_error_sum / sign_distance_count


def run_positioning_test(expected_metadata, actual_metadata):
    expected_dictionary = proto_api.create_images_dictionary(expected_metadata)
    actual_dictionary = proto_api.create_images_dictionary(actual_metadata)
    average_error_meters = get_positioning_average_error_meters(expected_dictionary, actual_dictionary)
    return average_error_meters


class SignPositioningPerformanceTest(AbstractBaseTest, unittest.TestCase):
    ftp_path = "/ORBB/data/test/python/sign_positioning/positioning_test_data.zip"
    resources_folder = "performance"
    max_average_error = 9.60

    def test_performance(self):
        data_path = os.path.join(resource_utils.LOCAL_TEST_RESOURCES_FOLDER, self.resources_folder, "positioning_test_data")
        input_file_path = os.path.join(data_path, "input.bin")
        ground_truth_file_path = os.path.join(data_path, "localized_gt.bin")
        output_file_path = os.path.join(data_path, "output.bin")
        first_local_app_path = "../../sign_positioning/tools/first_local_app/"
        first_local_runner = FirstLocalRunner(first_local_app_path, input_file_path, output_file_path)
        first_local_runner()
        expected_metadata = proto_api.read_imageset_file(ground_truth_file_path)
        actual_metadata = proto_api.read_imageset_file(output_file_path)
        average_error_metters = run_positioning_test(expected_metadata, actual_metadata)
        print("Average error in meters {:.2f} m".format(average_error_metters))
        assert( average_error_metters <= self.max_average_error )


if __name__ == '__main__':
    log_util.config(__file__)
    unittest.main()

