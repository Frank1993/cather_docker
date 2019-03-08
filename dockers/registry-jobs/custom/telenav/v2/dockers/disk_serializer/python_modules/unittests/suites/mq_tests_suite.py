import sys
import os
import logging

sys.path.append(os.path.abspath('../../'))
sys.path.append(os.path.abspath('../../apollo_python_common/protobuf'))
sys.path.append(os.path.abspath('../../ocr/attention_ocr/'))
sys.path.append(os.path.abspath('../../ocr/attention_ocr/datasets'))
import unittest
import tensorflow as tf

import apollo_python_common.log_util as log_util
from unittests.mq.quality_orientation_selector.quality_orientation_selector_test import QualityOrientationSelectorTest
from unittests.mq.geo_selector.geo_selector_test import GeoSelectorTest
from unittests.mq.classif_predictor.classif_predictor_test import ClassifPredictorTest
from unittests.mq.ocr.ocr_test import OCRTest
from unittests.mq.signs_facing.signs_facing_test import SignsFacingTest

# from unittests.mq.retinanet.retinanet_predictor_test import RetinanetPredictorTest


def run_tests_with_tf_session(test_files):
    runner = unittest.TextTestRunner(verbosity=2)
    for test_file in test_files:
        graph = tf.Graph()
        with graph.as_default():
            session = tf.Session()
            with session.as_default():
                runner.run(unittest.TestLoader().loadTestsFromTestCase(test_file))


def run_tests(test_files):
    runner = unittest.TextTestRunner(verbosity=2)
    for test_file in test_files:
        runner.run(unittest.TestLoader().loadTestsFromTestCase(test_file))


if __name__ == '__main__':
    log_util.config(__file__)
    logger = logging.getLogger(__name__)

    tf_test_files = [
       OCRTest,
       GeoSelectorTest,
       QualityOrientationSelectorTest,
       ClassifPredictorTest
        #         RetinanetPredictorTest
    ]

    test_files = [
        SignsFacingTest
    ]

    try:
        run_tests_with_tf_session(tf_test_files)
        run_tests(test_files)
    except Exception as err:
        logger.error(err, exc_info=True)
        raise err
