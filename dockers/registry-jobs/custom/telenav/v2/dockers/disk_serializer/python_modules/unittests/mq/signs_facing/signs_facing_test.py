import os
import shutil
import threading
import unittest
from time import sleep

from apollo_python_common import io_utils, proto_api
from apollo_python_common.rectangle import Rectangle
from apollo_python_common.ml_pipeline.config_api import MQ_Param
from classification.fast_ai.signs_facing_classifier.mq.signs_facing_mq_predictor import SignsFacingMQPredictor
from unittests.mq.components.accumulation_consumer import AccumulationConsumer
from unittests.mq.components.image_proto_mq_provider import MQProviderFromImageSet
from unittests.utils import resource_utils
from unittests.utils.resource_utils import LOCAL_TEST_RESOURCES_FOLDER


class SignsFacingTest(unittest.TestCase):
    GENERIC_CONFIG_PATH = "../mq/configs/generic_config.json"
    EXPECTED_SCORE = 27
    IOU_THRESHOLD = 0.25
    ftp_bundle_path = "/ORBB/data/test/python/mq/classif/signs_facing/mq_test_bundle.zip"
    local_download_folder = "signs_facing"
    local_bundle_folder = "signs_facing/mq_test_bundle"

    def _get_predictor(self, input_queue, output_queue, model_dir):
        conf = io_utils.json_load(self.GENERIC_CONFIG_PATH)
        conf[MQ_Param.MQ_INPUT_QUEUE_NAME] = input_queue
        conf[MQ_Param.MQ_OUTPUT_QUEUE_NAME] = output_queue

        conf["model_dir"] = model_dir
        conf["model_name"] = "2018_12_17_13_20_final"
        conf["label_list_file"] = "models/2018_12_17_13_20_labels.json"
        conf["backbone_model"] = "resnet50"
        conf["batch_size"] = 256
        conf["image_size"] = 84
        conf["sq_crop_factor"] = 1.2
        conf["tfms_max_rotate"] = 0.0
        conf["tfms_max_warp"] = 0
        conf["tfms_flip_vert"] = False
        conf["tfms_do_flip"] = False

        return SignsFacingMQPredictor(conf)

    def _get_provider(self, output_queue, gt_rois_file):
        conf = io_utils.json_load(self.GENERIC_CONFIG_PATH)
        conf[MQ_Param.MQ_OUTPUT_QUEUE_NAME] = output_queue
        return MQProviderFromImageSet(conf, gt_rois_file)

    def _get_accumulator(self, input_queue):
        conf = io_utils.json_load(self.GENERIC_CONFIG_PATH)
        conf[MQ_Param.MQ_INPUT_QUEUE_NAME] = input_queue

        return AccumulationConsumer(conf)

    @staticmethod
    def _is_same_roi(gt_roi, pred_roi):
        gt_rect = Rectangle(gt_roi.rect.tl.col, gt_roi.rect.tl.row, gt_roi.rect.br.col, gt_roi.rect.br.row)
        pred_rect = Rectangle(pred_roi.rect.tl.col, pred_roi.rect.tl.row, pred_roi.rect.br.col, pred_roi.rect.br.row)

        return gt_rect.intersection_over_union(pred_rect) >= SignsFacingTest.IOU_THRESHOLD

    @staticmethod
    def _create_pred_imageset(accumulator):
        accumulated_protos = accumulator.get_accumulated_protos()
        pred_imageset = proto_api.get_new_imageset_proto()
        for image_proto in accumulated_protos:
            new_image_proto = pred_imageset.images.add()
            new_image_proto.CopyFrom(image_proto)

        return pred_imageset

    def _compute_model_score(self, pred_imageset, gt_imageset):
        pred_dict = proto_api.create_images_dictionary(pred_imageset)
        gt_dict = proto_api.create_images_dictionary(gt_imageset)

        ok_left = 0
        ok_right = 0
        mc_front = 0
        for img_name, pred_rois in pred_dict.items():
            gt_rois = gt_dict[img_name]

            for pred_roi in pred_rois:
                matched_gt_rois = [gt_roi for gt_roi in gt_rois if self._is_same_roi(gt_roi, pred_roi)]
                if len(matched_gt_rois) == 1:
                    matched_gt_roi = matched_gt_rois[0]
                    if matched_gt_roi.local.angle_of_roi == 0 and pred_roi.local.angle_of_roi != 0:
                        mc_front += 1
                    elif matched_gt_roi.local.angle_of_roi == -1 and pred_roi.local.angle_of_roi == -1:
                        ok_left += 1
                    elif matched_gt_roi.local.angle_of_roi == 1 and pred_roi.local.angle_of_roi == 1:
                        ok_right += 1
                else:
                    raise IndexError("more than one ground truth matching the predicted ROI: ".format(matched_gt_rois))

        return ok_left + ok_right - mc_front

    def setUp(self):
        print("Downloading resources from FTP...")
        resource_utils.ensure_test_resource(self.ftp_bundle_path, self.local_download_folder)

    def tearDown(self):
        print("Cleaning up imgs resources...")
        if os.path.exists(LOCAL_TEST_RESOURCES_FOLDER):
            shutil.rmtree(LOCAL_TEST_RESOURCES_FOLDER)

    def test_signs_facing(self):
        ocr_input_queue_name = io_utils.get_random_file_name()
        ocr_output_queue_name = io_utils.get_random_file_name()

        model_dir = os.path.join(LOCAL_TEST_RESOURCES_FOLDER, self.local_bundle_folder)
        gt_rois_file = os.path.join(LOCAL_TEST_RESOURCES_FOLDER, self.local_bundle_folder, "imgs", "rois.bin")

        provider = self._get_provider(ocr_input_queue_name, gt_rois_file)
        predictor = self._get_predictor(ocr_input_queue_name, ocr_output_queue_name, model_dir)
        accumulator = self._get_accumulator(ocr_output_queue_name)

        provider_thread = threading.Thread(target=lambda: provider.start())
        provider_thread.daemon = True
        provider_thread.start()

        ocr_thread = threading.Thread(target=lambda: predictor.start())
        ocr_thread.daemon = True
        ocr_thread.start()

        acc_thread = threading.Thread(target=lambda: accumulator.start())
        acc_thread.daemon = True
        acc_thread.start()

        while len(accumulator.get_accumulated_protos()) < provider.get_proto_list_size():
            sleep(1)

        pred_imageset = self._create_pred_imageset(accumulator)
        gt_imageset = proto_api.read_imageset_file(gt_rois_file)

        score = self._compute_model_score(pred_imageset, gt_imageset)
        print("Score ", score)
        assert (score == self.EXPECTED_SCORE)

        provider.delete_queue(ocr_input_queue_name)
        provider.delete_queue(ocr_output_queue_name)
