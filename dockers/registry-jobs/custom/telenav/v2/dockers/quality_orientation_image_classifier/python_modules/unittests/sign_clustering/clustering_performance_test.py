import os
import pandas as pd
import unittest

import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util

from sign_clustering import clustering
from sign_clustering.evaluate_clusters import compare_clusters_to_gt
from unittests.common.abstract_base_test import AbstractBaseTest
from unittests.utils import resource_utils as resource_utils


class ClusteringPerformanceTest(AbstractBaseTest, unittest.TestCase):
    ftp_path = "/ORBB/data/test/python/clustering/clustering_data.zip"
    resources_folder = "performance"
    expected_precision = 0.89

    def test_performance(self):
        data_path = os.path.join(resource_utils.LOCAL_TEST_RESOURCES_FOLDER, self.resources_folder, "clustering_data")

        input_file = os.path.join(data_path, "localization.bin")
        config_file = "../../sign_clustering/config/clustering_config.json"
        ground_truth_file = os.path.join(data_path, "extended_clusters.csv")

        config = io_utils.config_load(config_file)
        rois_df = clustering.read_detections_input(input_file, config)
        gt_df = pd.read_csv(ground_truth_file)

        merged_df = pd.merge(gt_df, rois_df)
        merged_df = clustering.filter_detections_by_distance(merged_df, config.roi_distance_threshold)
        merged_df = clustering.filter_detections_by_gps_acc(merged_df, config.gps_accuracy_threshold)

        nr_clusters, clustered_df = clustering.get_clusters(merged_df, config, threads_number=1)
        completeness_score, precision, recall, accuracy, distance_avg = compare_clusters_to_gt(clustered_df)

        print("Precision = {}".format(precision))
        assert (precision >= self.expected_precision)


if __name__ == '__main__':
    log_util.config(__file__)
    unittest.main()
