import logging
import os
import shutil
import threading
import unittest
from time import sleep

import apollo_python_common.io_utils as io_utils
import apollo_python_common.proto_api as proto_api
from apollo_python_common.generate_model_statistics import get_statistics
from object_detection.retinanet.mq.predictor import RetinaNetPredictor
from unittests.mq.components.accumulation_consumer import AccumulationConsumer
from unittests.mq.retinanet.custom_mq_predictor import CustomMQProvider
from unittests.utils import resource_utils


class RetinanetPredictorTest(unittest.TestCase):
    local_imgs_folder = "imgs/"
    local_bundle_folder = "bundle/"
    generic_config_path = "../mq/configs/generic_config.json"

    def get_provider(self, dataset_path, output_queue):
        conf = io_utils.json_load(self.generic_config_path)
        conf["mq_output_queue_name"] = output_queue

        return CustomMQProvider(conf, dataset_path)

    def get_accumulation_consumer(self, queue):
        conf = io_utils.json_load(self.generic_config_path)
        conf["mq_input_queue_name"] = queue

        return AccumulationConsumer(conf)

    def get_predictor(self, input_queue, output_queue, bundle_path):
        conf = io_utils.json_load(self.generic_config_path)
        conf["mq_input_queue_name"] = input_queue
        conf["mq_output_queue_name"] = output_queue

        conf["weights_file"] = os.path.join(bundle_path, "retinanet_resnet50_traffic_signs_v002.pb")
        conf["score_thresholds_file"] = os.path.join(bundle_path, "classes_thresholds.json")
        conf["train_meta_file"] = os.path.join(bundle_path, "rois_train.bin")

        conf["multi_gpu"] = 1
        conf["algorithm"] = "RetinaNet"
        conf["algorithm_version"] = "-1"
        conf["predict_min_side_size"] = 20

        return RetinaNetPredictor(conf)

    def imge_proto_list_2_imageset_proto(self, img_proto_list):
        imageset_proto = proto_api.get_new_imageset_proto()

        for img_proto in img_proto_list:
            new_image_proto = imageset_proto.images.add()
            new_image_proto.CopyFrom(img_proto)

        return imageset_proto

    def compute_accuracy(self, pred_img_proto_list, dataset_path):

        gt_proto_path = os.path.join(dataset_path, "rois.bin")
        pred_imageset_proto = self.imge_proto_list_2_imageset_proto(pred_img_proto_list)
        gt_imageset_proto = proto_api.read_imageset_file(gt_proto_path)

        for img_proto in gt_imageset_proto.images:
            img_proto.metadata.image_path = os.path.join(dataset_path, img_proto.metadata.image_path)

        statistics_dict = get_statistics(gt_imageset_proto, pred_imageset_proto)
        acc = statistics_dict["Total"].accuracy()

        return acc

    def perform_test(self, dataset_path, bundle_path):
        logger = logging.getLogger(__name__)

        input_queue = io_utils.get_random_file_name()
        pred_queue = io_utils.get_random_file_name()

        provider = self.get_provider(dataset_path, input_queue)
        predictor = self.get_predictor(input_queue, pred_queue, bundle_path)
        accumulator = self.get_accumulation_consumer(pred_queue)

        predictor_thread = threading.Thread(target=lambda: predictor.start())
        predictor_thread.daemon = True

        accumulator_thread = threading.Thread(target=lambda: accumulator.start())
        accumulator_thread.daemon = True

        provider.start()
        predictor_thread.start()
        accumulator_thread.start()

        logger.info("Waiting for accumulation of protos...")
        pred_img_proto_list = accumulator.get_accumulated_protos()
        while len(pred_img_proto_list) < provider.get_proto_list_size():
            sleep(1)
            pred_img_proto_list = accumulator.get_accumulated_protos()

        acc = self.compute_accuracy(pred_img_proto_list, dataset_path)

        provider.delete_queue(input_queue)
        provider.delete_queue(pred_queue)

        return acc

    def generic_retina_test(self, imgs_ftp_path, bundle_ftp_path, min_performance):
        logger = logging.getLogger(__name__)

        local_imgs_path = os.path.join(resource_utils.LOCAL_TEST_RESOURCES_FOLDER, self.local_imgs_folder)
        local_bundle_path = os.path.join(resource_utils.LOCAL_TEST_RESOURCES_FOLDER, self.local_bundle_folder)

        try:
            self.set_up_environment(imgs_ftp_path, local_imgs_path, bundle_ftp_path, local_bundle_path)

            acc = self.perform_test(local_imgs_path, local_bundle_path)

            assert (acc >= min_performance)

        except Exception as e:
            logger.error(e)

    def set_up_environment(self, imgs_ftp_path, local_imgs_path, bundle_ftp_path, local_bundle_path):
        logger = logging.getLogger(__name__)

        self.tear_down_environment(local_imgs_path, local_bundle_path)

        logger.info("Downloading resources from FTP...")
        resource_utils.ensure_test_resource(imgs_ftp_path, self.local_imgs_folder)
        resource_utils.ensure_test_resource(bundle_ftp_path, self.local_bundle_folder)

    def tear_down_environment(self, local_imgs_path, local_bundle_path):
        logger = logging.getLogger(__name__)

        if os.path.exists(local_imgs_path) and os.path.isdir(local_imgs_path):
            logger.info("Cleaning up imgs resources...")
            shutil.rmtree(local_imgs_path)

        if os.path.exists(local_bundle_path) and os.path.isdir(local_bundle_path):
            logger.info("Cleaning up bundle resources...")
            shutil.rmtree(local_bundle_path)

    def test_1(self):

        imgs_ftp_path = "test/python/mq/retinanet/unittests_imgs.zip"
        bundle_ftp_path = "test/python/mq/retinanet/retinanet_weights.zip"

        min_performance = 0.8
        self.generic_retina_test(imgs_ftp_path, bundle_ftp_path, min_performance)
