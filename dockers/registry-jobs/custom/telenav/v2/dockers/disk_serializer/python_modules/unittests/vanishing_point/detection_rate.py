import os
import sys
sys.path.append(os.path.abspath('../../'))
sys.path.append(os.path.abspath('../../common/protobuf/'))
from collections import namedtuple
import zipfile

import cv2

import apollo_python_common.proto_api as roi_metadata
from vanishing_point.vanishing_point import VanishingPointDetector
import test.utils.resource_utils as res_utils

TEST_FOLDER = "vp_markings"

# we consider a difference of 3% of the image size an acceptable error
ERROR_THRESHOLD = 0.03

VPDetection = namedtuple('VPDetection', 'x y confidence')


def get_test_set():
    test_resources_path = res_utils.LOCAL_TEST_RESOURCES_FOLDER + "/" + TEST_FOLDER
    if not os.path.isdir(test_resources_path):
        full_local_path = res_utils.copy_test_file_from_ftp_to_local('python/vp_markings.zip', '/')
        with zipfile.ZipFile(full_local_path, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(full_local_path))
        os.remove(full_local_path)
    return test_resources_path


def get_statistics(results):
    correct_pitch_count = 0
    correct_yaw_count = 0
    for result in results:
        difference_threshold = result[2] * ERROR_THRESHOLD
        if abs(result[0].x - result[1].x) < difference_threshold:
            correct_yaw_count += 1
        if abs(result[0].y - result[1].y) < difference_threshold:
            correct_pitch_count += 1
    pitch_rate = correct_pitch_count / len(results)
    yaw_rate = correct_yaw_count / len(results)
    print("pitch rate: {pitch_rate} yaw rate: {yaw_rate}".format(pitch_rate=pitch_rate, yaw_rate=yaw_rate))


if __name__ == "__main__":
    vp_detector = VanishingPointDetector()
    test_set_path = get_test_set()
    metadata = roi_metadata.read_imageset_file(test_set_path + "/rois.bin")
    results = []
    for image_entry in metadata.image_rois:
        if image_entry.HasField('vanishing_point'):
            image_filename = image_entry.file
            marked_vp = image_entry.vanishing_point
            marked_vp = VPDetection(marked_vp.vp.col, marked_vp.vp.row, marked_vp.confidence)
            img = cv2.imread(test_set_path + "/" + image_filename, cv2.IMREAD_COLOR)
            detected_vp, confidence = vp_detector.get_vanishing_point(img)
            detected_vp = VPDetection(int(detected_vp.x), int(detected_vp.y), confidence)
            reference_size = max(img.shape[0], img.shape[1])
            results.append((marked_vp, detected_vp, reference_size))
    get_statistics(results)
