import sys
import os
import logging

sys.path.append(os.path.abspath('../../'))
sys.path.append(os.path.abspath('../../apollo_python_common/protobuf'))
sys.path.append(os.path.abspath('../../ocr/attention_ocr/'))
sys.path.append(os.path.abspath('../../ocr/attention_ocr/datasets'))

import unittest
import apollo_python_common.log_util as log_util

import tensorflow as tf

from unittests.sign_facing_classification.signs_facing_classifier_performance_test import \
    SignsFacingClassifierPerformanceTest
from unittests.image_orientation.filter_bad_orientation_images_test import FilterBadOrientationImagesTest
from unittests.image_orientation.image_orientation_performance_test import ImageOrientationPerformanceTest
from unittests.image_quality.image_quality_performance_test import ImageQualityPerformanceTest
from unittests.roi_classifier.roi_classifier_performance_test import RoiClassifierPerformanceTest
from unittests.sign_clustering.clustering_performance_test import ClusteringPerformanceTest
from unittests.ocr.ocr_performance_test import OCRPerformanceTest

def run_tests(test_files):
    
    runner = unittest.TextTestRunner(verbosity=2)    
    for test_file in test_files:
        graph = tf.Graph()
        with graph.as_default():
            session = tf.Session()
            with session.as_default():
                runner.run(unittest.TestLoader().loadTestsFromTestCase(test_file))


if __name__ == '__main__':
    log_util.config(__file__)
    logger = logging.getLogger(__name__)

    test_files = [
        ImageOrientationPerformanceTest,
        FilterBadOrientationImagesTest,
        ImageQualityPerformanceTest,
        OCRPerformanceTest,
        SignsFacingClassifierPerformanceTest,
        RoiClassifierPerformanceTest,
        ClusteringPerformanceTest
    ]

    try:
        run_tests(test_files)
                    
    except Exception as err:
        logger.error(err, exc_info=True)
        raise err
