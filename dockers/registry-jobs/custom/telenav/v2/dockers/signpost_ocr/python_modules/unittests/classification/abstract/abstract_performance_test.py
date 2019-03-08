import os

import classification.scripts.validator as validator
import pandas as pd
from classification.scripts.constants import Column, INVALID_IMG_PRED
from classification.scripts.prediction.folder_predictor import FolderPredictor
from unittests.common.abstract_base_test import AbstractBaseTest
from unittests.utils import resource_utils as resource_utils


class AbstractPerformanceTest(AbstractBaseTest):
    resources_folder = "performance"
    ftp_path = None
    ftp_bundle_path = None
    expected_acc = None

    def post_process_predictions(self, pred_data_df, predictor_params, label_class=None):
        """ This method should be overwritten in the implementations of this test that need to do some post
         processing on the prediction data frame before computing the accuracy.
         :param pred_data_df: the predictions data frame resulting from the FolderPredictor
         :param label_class: the label class folder name for which the predictions are currently run
         :param predictor_params: the parameters with which the predictor was run """

        pred_data_df = pred_data_df[pred_data_df[Column.PRED_CLASS_COL] != INVALID_IMG_PRED]
        pred_data_df.loc[:, 'correct'] = pred_data_df.loc[:, Column.PRED_CLASS_COL].apply(
            lambda pred_class: pred_class == label_class)

        return pred_data_df

    def check_performance_metric(self, combined_pred_data_df):
        """ In case we need to test a different metric than accuracy in the implementing test cases, we just need to
         override this method. """

        acc = validator.compute_accuracy(combined_pred_data_df)
        acc = round(acc, 3)

        print("Accuracy = {}".format(acc))
        assert (acc == self.expected_acc)

    def test_performance(self):
        dataset_path = os.path.join(resource_utils.LOCAL_TEST_RESOURCES_FOLDER, self.resources_folder, "raw_imgs")

        predictor = FolderPredictor(self.ftp_bundle_path)

        combined_pred_data_df_list = []

        for label_class in os.listdir(dataset_path):
            label_class_path = os.path.join(dataset_path, label_class)

            pred_data_df = predictor.compute_prediction(label_class_path)
            pred_data_df = self.post_process_predictions(pred_data_df, predictor.params, label_class=label_class)

            combined_pred_data_df_list.append(pred_data_df)

        combined_pred_data_df = pd.concat(combined_pred_data_df_list)

        self.check_performance_metric(combined_pred_data_df)
