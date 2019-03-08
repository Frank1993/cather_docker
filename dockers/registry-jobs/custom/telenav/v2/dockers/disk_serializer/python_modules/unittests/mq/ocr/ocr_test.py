import os

import shutil
from tqdm import tqdm
import threading
from time import sleep

tqdm.pandas()

import unittest

import apollo_python_common.io_utils as io_utils
import apollo_python_common.proto_api as proto_api

from apollo_python_common.ml_pipeline.config_api import MQ_Param
from ocr.scripts.prediction.mq_ocr_predictor import OCR_MQ_Predictor
from unittests.mq.components.accumulation_consumer import AccumulationConsumer

from unittests.mq.components.image_proto_mq_provider import MQProviderFromImageSet

from unittests.utils import resource_utils
from unittests.utils.resource_utils import LOCAL_TEST_RESOURCES_FOLDER


class OCRTest(unittest.TestCase):
    GENERIC_CONFIG_PATH = "../mq/configs/generic_config.json"
    MIN_COMPONENT_SIZE = 25
    CONF_THRESH = 0.85
    DATASET_NAME = "fake_dataset"
    EXPECTED_ACCURACY = 0.9209

    ftp_imgs_path = "/ORBB/data/test/python/mq/ocr/mq_imgs.zip"
    local_imgs_folder = "imgs/"

    ftp_ckpt_path = "/ORBB/data/test/python/ocr/ocr_ckpt.zip"
    local_ckpt_folder = "ckpt/"

    def get_provider(self, output_queue, gt_rois_file):
        conf = io_utils.json_load(self.GENERIC_CONFIG_PATH)
        conf[MQ_Param.MQ_OUTPUT_QUEUE_NAME] = output_queue
        return MQProviderFromImageSet(conf, gt_rois_file)

    def get_predictor(self, input_queue, output_queue, ckpt_path):
        conf = io_utils.json_load(self.GENERIC_CONFIG_PATH)
        conf[MQ_Param.MQ_INPUT_QUEUE_NAME] = input_queue
        conf[MQ_Param.MQ_OUTPUT_QUEUE_NAME] = output_queue

        conf["min_component_size"] = self.MIN_COMPONENT_SIZE
        conf["conf_thresh"] = self.CONF_THRESH
        conf["ckpt_path"] = ckpt_path
        conf["dataset"] = self.DATASET_NAME

        return OCR_MQ_Predictor(conf)

    def get_accumulator(self, input_queue):
        conf = io_utils.json_load(self.GENERIC_CONFIG_PATH)
        conf[MQ_Param.MQ_INPUT_QUEUE_NAME] = input_queue

        return AccumulationConsumer(conf)

    def create_pred_imageset(self, accumulator):
        accumulated_protos = accumulator.get_accumulated_protos()
        pred_imageset = proto_api.get_new_imageset_proto()
        for image_proto in accumulated_protos:
            new_image_proto = pred_imageset.images.add()
            new_image_proto.CopyFrom(image_proto)

        return pred_imageset

    def get_text_components(self, roi):
        return [c for c in roi.components if
                proto_api.get_component_type_name(c.type) == "GENERIC_TEXT"]

    def create_comp_dict(self, imageset):
        roi_dict = proto_api.create_images_dictionary(imageset)

        component_dict = {}

        for img_name, roi_list in list(roi_dict.items()):
            signpost_rois = [roi for roi in roi_list if proto_api.get_roi_type_name(roi.type) == "SIGNPOST_GENERIC"]
            all_text_components = []
            for roi in signpost_rois:
                text_components = self.get_text_components(roi)
                text_components = [t for t in text_components if len(t.value) != 0]
                all_text_components += text_components

            component_dict[img_name] = all_text_components

        return component_dict

    def is_same_comp(self, gt_comp, pred_comp):
        return gt_comp.box.tl.row == pred_comp.box.tl.row and \
               gt_comp.box.tl.col == pred_comp.box.tl.col and \
               gt_comp.box.br.row == pred_comp.box.br.row and \
               gt_comp.box.br.col == pred_comp.box.br.col

    def compute_accuracy(self, gt_imageset, pred_imageset):

        gt_comp_dict = self.create_comp_dict(gt_imageset)
        pred_comp_dict = self.create_comp_dict(pred_imageset)

        nr_pred_texts, nr_correct_texts = 0, 0

        for img_name, pred_comps in list(pred_comp_dict.items()):
            gt_comps = gt_comp_dict[img_name]
            nr_pred_texts += len(pred_comps)
            for pred_comp in pred_comps:
                matched_gt_comp_list = [gt_comp for gt_comp in gt_comps if self.is_same_comp(gt_comp, pred_comp)]
                if len(matched_gt_comp_list) == 1:
                    matched_gt_comp = matched_gt_comp_list[0]
                    pred_text = pred_comp.value
                    gt_text = matched_gt_comp.value

                    nr_correct_texts += 1 if pred_text == gt_text else 0

        return round(nr_correct_texts / nr_pred_texts, 4)

    def setUp(self):
        print("Downloading resources from FTP...")
        resource_utils.ensure_test_resource(self.ftp_imgs_path, self.local_imgs_folder)
        resource_utils.ensure_test_resource(self.ftp_ckpt_path, self.local_ckpt_folder)

    def tearDown(self):
        print("Cleaning up imgs resources...")
        if os.path.exists(LOCAL_TEST_RESOURCES_FOLDER):
            shutil.rmtree(LOCAL_TEST_RESOURCES_FOLDER)

    def test_ocr(self):

        ocr_input_queue_name = io_utils.get_random_file_name()
        ocr_output_queue_name = io_utils.get_random_file_name()

        full_ckpt_path = os.path.join(LOCAL_TEST_RESOURCES_FOLDER, self.local_ckpt_folder, "model.ckpt-71251")

        gt_rois_file = os.path.join(LOCAL_TEST_RESOURCES_FOLDER, self.local_imgs_folder, "mq_imgs", "rois.bin")

        provider = self.get_provider(ocr_input_queue_name, gt_rois_file)
        predictor = self.get_predictor(ocr_input_queue_name, ocr_output_queue_name, full_ckpt_path)
        accumulator = self.get_accumulator(ocr_output_queue_name)

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

        pred_imageset = self.create_pred_imageset(accumulator)
        gt_imageset = proto_api.read_imageset_file(gt_rois_file)

        accuracy = self.compute_accuracy(gt_imageset, pred_imageset)

        print("Accuracy ", accuracy)
        assert (accuracy == self.EXPECTED_ACCURACY)

        provider.delete_queue(ocr_input_queue_name)
        provider.delete_queue(ocr_output_queue_name)
