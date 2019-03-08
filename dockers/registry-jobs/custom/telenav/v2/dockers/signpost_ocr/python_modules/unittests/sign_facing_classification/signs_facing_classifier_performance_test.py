import os
import unittest

import classification.scripts.signs_facing_classifier.postprocess as pp
from classification.scripts.signs_facing_classifier.constants import PredDfColumn
from apollo_python_common import io_utils, ftp_utils
from unittests.classification.abstract.abstract_performance_test import AbstractPerformanceTest
from unittests.utils import resource_utils


class SignsFacingClassifierPerformanceTest(AbstractPerformanceTest, unittest.TestCase):
    ftp_path = "/ORBB/data/test/python/classification/sign_facing/unittests_imgs.zip"
    ftp_bundle_path = "/ORBB/data/roi_angle_predictor/signs_facing_classifier.zip"
    expected_score = 21

    def setUp(self):
        super().setUp()
        self.lt, self.rt = self.__load_best_thresholds()

    def __load_best_thresholds(self):
        """ Loads the best thresholds computed for the current model, these will be used to update predictions, by
        changing the prediction value based on the threshold value. """
        local_resources_path = os.path.join(resource_utils.LOCAL_TEST_RESOURCES_FOLDER, self.resources_folder)
        ftp_utils.copy_zip_and_extract(self.ftp_bundle_path, local_resources_path)

        path = os.path.join(local_resources_path, "model_best_thresholds.json")

        return pp.load_best_thresholds_json(path)

    def post_process_predictions(self, pred_data_df, predictor_params, label_class=None):
        pred_df = pp.get_pred_confidence_df(pred_data_df, predictor_params)
        pred_df.loc[:, PredDfColumn.GT_CLASS_COL] = label_class
        pred_df = pp.apply_thresholds(pred_df, self.lt, self.rt)

        return pred_df

    def check_performance_metric(self, combined_pred_data_df):
        # print("check perf metric combined pred data head: \n", combined_pred_data_df.head())
        score = pp.compute_model_score(combined_pred_data_df)

        print("Model Score = {}".format(score))
        assert (score == self.expected_score)
