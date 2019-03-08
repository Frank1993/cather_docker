import logging
import os
import threading
import unittest
from time import sleep

import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
from apollo_python_common.proto_api import MQ_Messsage_Type
from apollo_python_common.ml_pipeline.config_api import MQ_Param

from sign_clustering.mq.predictor import SignClusteringPredictor
from sign_clustering.evaluate_clusters import compare_cluster_proto_to_gt_file
from unittests.common.abstract_base_test import AbstractBaseTest
from unittests.mq.components.accumulation_consumer import AccumulationConsumer
from unittests.mq.traffic_signs_aggregator.clustering_mq_provider import ClusteringMQProvider
from unittests.utils import resource_utils


class ClusteringMQTest(AbstractBaseTest, unittest.TestCase):
    ftp_path = "/ORBB/data/test/python/clustering/clustering_data.zip"
    resources_folder = "mq"
    expected_precision = 0.89
    config_path = "../../../sign_clustering/config/clustering_config.json"

    def get_provider(self, output_queue, clustering_data_path):
        conf = io_utils.config_load(self.config_path)
        conf[MQ_Param.MQ_OUTPUT_QUEUE_NAME] = output_queue

        return ClusteringMQProvider(conf, clustering_data_path)

    def get_predictor(self, input_queue, output_queue):
        conf = io_utils.config_load(self.config_path)
        conf[MQ_Param.MQ_INPUT_QUEUE_NAME] = input_queue
        conf[MQ_Param.MQ_OUTPUT_QUEUE_NAME] = output_queue

        return SignClusteringPredictor(conf)

    def get_accumulation_consumer(self, queue):
        conf = io_utils.config_load(self.config_path)
        conf[MQ_Param.MQ_INPUT_QUEUE_NAME] = queue

        return AccumulationConsumer(conf, mq_message_type=MQ_Messsage_Type.GEO_TILE)

    def perform_test(self, local_imageset_path, local_gt_path):
        logger = logging.getLogger(__name__)

        provider_queue = io_utils.get_random_file_name()
        clustering_queue = io_utils.get_random_file_name()

        provider = self.get_provider(provider_queue, local_imageset_path)
        provider.start()

        predictor = self.get_predictor(provider_queue, clustering_queue)
        predictor_thread = threading.Thread(target=lambda: predictor.start())
        predictor_thread.daemon = True

        accumulator = self.get_accumulation_consumer(clustering_queue)
        accumulator_thread = threading.Thread(target=lambda: accumulator.start())
        accumulator_thread.daemon = True

        predictor_thread.start()
        accumulator_thread.start()

        logger.info("Waiting for accumulation of protos...")
        proto_list = accumulator.get_accumulated_protos()
        while len(proto_list) < provider.get_proto_list_size():
            sleep(1)
            proto_list = accumulator.get_accumulated_protos()

        geotile_proto = proto_list[0]
        completeness_score, precision, recall, accuracy, distance_avg = \
            compare_cluster_proto_to_gt_file(geotile_proto.clusters, local_gt_path)

        return precision

    def test_mq_clustering(self):
        logger = logging.getLogger(__name__)

        data_path = os.path.join(resource_utils.LOCAL_TEST_RESOURCES_FOLDER, self.resources_folder, "clustering_data")
        local_imageset_path = os.path.join(data_path, "localization.bin")
        local_gt_path = os.path.join(data_path, "extended_clusters.csv")

        try:
            precision = self.perform_test(local_imageset_path, local_gt_path)
            logger.info("PRECISION = {}".format(precision))
            assert (precision >= self.expected_precision)
        except Exception as e:
            logger.error(e)


if __name__ == '__main__':
    log_util.config(__file__)
    unittest.main()
