import logging
import os
import shutil
import threading
import unittest
from glob import glob
from time import sleep

import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
import pandas as pd
from classification.scripts.prediction.classif_mq_predictor import ClassifPredictor
from unittests.mq.classif_predictor.custom_mq_provider import CustomMQProvider
from unittests.mq.components.accumulation_consumer import AccumulationConsumer
from unittests.utils import resource_utils

import traceback

class ClassifPredictorTest(unittest.TestCase):
    local_imgs_folder = "imgs/"
    local_bundle_folder = "bundle/"
    generic_config_path = "../mq/configs/generic_config.json"

    resources_folder = "classif"
    ftp_path = None
    ftp_bundle_path = None
    
    def get_provider(self, output_queue, unittest_imgs_path):
        conf = io_utils.json_load(self.generic_config_path)
        conf["mq_output_queue_name"] = output_queue

        return CustomMQProvider(conf, unittest_imgs_path)

    def get_predictor(self, input_queue, output_queue, local_bundle_path,algorithm):
        conf = io_utils.json_load(self.generic_config_path)
        conf["mq_input_queue_name"] = input_queue
        conf["mq_output_queue_name"] = output_queue
        conf[f"{algorithm}_bundle_path"]= local_bundle_path
        conf["algorithm_version"]="-1"
            
        return ClassifPredictor(conf)

    def get_accumulation_consumer(self, queue):
        conf = io_utils.json_load(self.generic_config_path)
        conf["mq_input_queue_name"] = queue

        return AccumulationConsumer(conf)

    def pred_classes_2_pred(self, pred_classes_proto):
        name_2_conf_list = [(pred_class.class_name, pred_class.confidence) for pred_class in pred_classes_proto]
        name_2_conf_list = sorted(name_2_conf_list, key=lambda name2conf: name2conf[1])
        return name_2_conf_list[-1][0]

    def proto_list_2_df(self, proto_list):

        return pd.DataFrame({
            "img_path": [proto.metadata.image_path for proto in proto_list],
            "pred_class": [self.pred_classes_2_pred(proto.features.classif_predictions[0].pred_classes) for proto in
                           proto_list]
        })

    def read_ground_truth_df(self, gt_path):

        classes = os.listdir(gt_path)
        class_df_list = []

        for class_name in classes:
            img_paths = glob(os.path.join(gt_path, class_name, "*"))
            class_df = pd.DataFrame({
                "img_path": img_paths
            })
            class_df["gt_class"] = class_name
            class_df_list.append(class_df)

        return pd.concat(class_df_list)

    def set_up_environment(self, imgs_ftp_path, local_imgs_path, bundle_ftp_path, local_bundle_path):
        logger = logging.getLogger(__name__)

        logger.info("Downloading resources from FTP...")
        self.tear_down_environment(local_imgs_path, local_bundle_path)

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

    def perform_test(self, local_imgs_path, local_bundle_path, algorithm):
        logger = logging.getLogger(__name__)

        provider_queue = io_utils.get_random_file_name()
        classif_queue = io_utils.get_random_file_name()

        provider = self.get_provider(provider_queue, local_imgs_path)
        provider.start()

        predictor = self.get_predictor(provider_queue, classif_queue, local_bundle_path, algorithm)
        predictor_thread = threading.Thread(target=lambda: predictor.start())
        predictor_thread.daemon = True

        accumulator = self.get_accumulation_consumer(classif_queue)
        accumulator_thread = threading.Thread(target=lambda: accumulator.start())
        accumulator_thread.daemon = True

        predictor_thread.start()
        accumulator_thread.start()

        logger.info("Waiting for accumulation of protos...")
        img_proto_list = accumulator.get_accumulated_protos()
        while len(img_proto_list) < provider.get_proto_list_size():
            sleep(1)
            img_proto_list = accumulator.get_accumulated_protos()

        pred_df = self.proto_list_2_df(img_proto_list)
        gt_df = self.read_ground_truth_df(local_imgs_path)

        joined_df = pd.merge(pred_df, gt_df, left_on="img_path", right_on="img_path")

        joined_df.loc[:, "correct"] = joined_df.apply(lambda row: 1 if row["pred_class"] == row["gt_class"] else 0,
                                                      axis=1)

        accuracy = joined_df["correct"].mean()

        provider.delete_queue(provider_queue)
        provider.delete_queue(classif_queue)

        return round(accuracy, 3)


    def generic_classif_test(self, imgs_ftp_path, bundle_ftp_path, algorithm, performance):
        logger = logging.getLogger(__name__)

        local_imgs_path = os.path.join(resource_utils.LOCAL_TEST_RESOURCES_FOLDER, self.local_imgs_folder)
        local_bundle_path = os.path.join(resource_utils.LOCAL_TEST_RESOURCES_FOLDER, self.local_bundle_folder)

        try:
            self.set_up_environment(imgs_ftp_path, local_imgs_path, bundle_ftp_path, local_bundle_path)

            acc = self.perform_test(local_imgs_path, local_bundle_path, algorithm)

            logger.info("ACCURACY = {}".format(acc))
            assert (acc == performance)

        except Exception as e:
            traceback.print_exc()
            logger.error(e)

        self.tear_down_environment(local_imgs_path, local_bundle_path)

    def test_orientation_classif(self):
        imgs_ftp_path = "/ORBB/data/test/python/mq/classif/image_orientation/unittests_imgs.zip"
        bundle_ftp_path = "/ORBB/data/image_orientation/good_bundle.zip"
        algorithm = "image_orientation"
        
        performance = 1
        self.generic_classif_test(imgs_ftp_path, bundle_ftp_path, algorithm, performance)

    def test_quality_classif(self):
        imgs_ftp_path = "/ORBB/data/test/python/mq/classif/image_quality/unittests_imgs.zip"
        bundle_ftp_path = "/ORBB/data/image_quality/good_bundle.zip"
        algorithm = "image_quality"

        performance = 0.962
        self.generic_classif_test(imgs_ftp_path, bundle_ftp_path,algorithm, performance)


if __name__ == '__main__':
    log_util.config(__file__)
    unittest.main()
