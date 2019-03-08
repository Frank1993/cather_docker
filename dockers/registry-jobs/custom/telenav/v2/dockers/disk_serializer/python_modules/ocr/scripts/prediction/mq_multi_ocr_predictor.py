import argparse
import logging

import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm

tqdm.pandas()

import apollo_python_common.image as image_api
import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
import apollo_python_common.proto_api as proto_api
import apollo_python_common.ml_pipeline.config_api as config_api
from apollo_python_common.ml_pipeline.multi_threaded_predictor import MultiThreadedPredictor
from ocr.scripts.prediction.ocr_predictor import OCRPredictor
from ocr.scripts.prediction.ocr_mq_param import OCR_MQ_Param as mq_param
from ocr.scripts.text_correction.traffic_signs_text_corrector import TrafficSignsTextCorrector
from ocr.scripts.text_correction.signpost_text_corrector import SignpostTextCorrector


class Multi_OCR_MQ_Predictor(MultiThreadedPredictor):
    SIGNPOST_GENERIC_PROTO_NAME = "SIGNPOST_GENERIC"
    GENERIC_TEXT_COMPONENT_NAME = "GENERIC_TEXT"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.comp_config = config_api.get_config_param(mq_param.COMPONENTS, self.config, {})
        self.predictor_dict = self.__build_predictor_dict()
        self.target_roi_classes = self.__get_target_roi_classes()

    def __get_target_roi_classes(self):
        roi_classes_to_predict_path = config_api.get_config_param(mq_param.ROI_CLASSES_TO_PREDICT_PATH,
                                                                  self.comp_config[mq_param.ROIS], "")

        return set(pd.read_csv(roi_classes_to_predict_path, names=["class"])["class"].values)

    def __build_predictor_dict(self):
        signpost_comp_graph, roi_graph = tf.Graph(), tf.Graph()

        with signpost_comp_graph.as_default():
            signpost_comp_ocr = self.__build_signpost_comp_ocr()
        with roi_graph.as_default():
            roi_ocr = self.__build_roi_ocr()

        return {
            mq_param.ROIS: roi_ocr,
            mq_param.SIGNPOST_COMP: signpost_comp_ocr
        }

    def __build_signpost_comp_ocr(self):
        spell_checker_resources_path = config_api.get_config_param(mq_param.SPELL_CHECKER_PATH,
                                                                   self.comp_config[mq_param.SIGNPOST_COMP], None)
        text_corrector = None if spell_checker_resources_path is None \
            else SignpostTextCorrector(spell_checker_resources_path)

        return OCRPredictor(config_api.get_config_param(mq_param.DATASET,
                                                        self.comp_config[mq_param.SIGNPOST_COMP], ""),
                            config_api.get_config_param(mq_param.CKPT_PATH,
                                                        self.comp_config[mq_param.SIGNPOST_COMP], ""),
                            text_corrector)

    def __build_roi_ocr(self):
        text_correction_resources_path = config_api.get_config_param(mq_param.TEXT_CORRECTION_RESOURCES_PATH,
                                                                     self.comp_config[mq_param.ROIS], None)
        text_corrector = None if text_correction_resources_path is None \
            else TrafficSignsTextCorrector(text_correction_resources_path)

        return OCRPredictor(config_api.get_config_param(mq_param.DATASET,
                                                        self.comp_config[mq_param.ROIS], ""),
                            config_api.get_config_param(mq_param.CKPT_PATH,
                                                        self.comp_config[mq_param.ROIS], ""),
                            text_corrector)

    def __comp_2_id(self, comp_proto):
        return "{}-{}-{}-{}".format(comp_proto.box.tl.row,
                                    comp_proto.box.tl.col,
                                    comp_proto.box.br.row,
                                    comp_proto.box.br.col)

    def __extract_cropped_comp(self, full_img, comp_proto):
        comp_id = self.__comp_2_id(comp_proto)
        comp_img = full_img[comp_proto.box.tl.row:comp_proto.box.br.row,
                   comp_proto.box.tl.col:comp_proto.box.br.col]

        width = self.predictor_dict[mq_param.SIGNPOST_COMP].width
        height = self.predictor_dict[mq_param.SIGNPOST_COMP].height

        comp_img = image_api.cv_resize(comp_img, width, height)
        return comp_id, comp_img

    def __get_signpost_rois(self, image_proto):
        return [roi for roi in image_proto.rois if
                proto_api.get_roi_type_name(roi.type) == self.SIGNPOST_GENERIC_PROTO_NAME]

    def __get_text_components(self, roi):
        return [c for c in roi.components if
                proto_api.get_component_type_name(c.type) == self.GENERIC_TEXT_COMPONENT_NAME]

    def __filter_components_by_size(self, text_components):
        min_component_size = self.comp_config[mq_param.SIGNPOST_COMP][mq_param.MIN_COMPONENT_SIZE]
        return [c for c in text_components if c.box.br.row - c.box.tl.row >= min_component_size]

    def __get_signpost_comp_preprocess_data(self, image_proto, img):

        signpost_rois = self.__get_signpost_rois(image_proto)

        id_2_components = {}
        for roi in signpost_rois:
            text_components = self.__get_text_components(roi)
            text_components = self.__filter_components_by_size(text_components)
            for comp_proto in text_components:
                comp_id, comp_img = self.__extract_cropped_comp(img, comp_proto)
                id_2_components[comp_id] = comp_img

        return id_2_components

    def __roi_2_id(self, roi):
        return "{}-{}-{}-{}".format(roi.rect.tl.row,
                                    roi.rect.tl.col,
                                    roi.rect.br.row,
                                    roi.rect.br.col)

    def __extract_cropped_roi(self, full_img, roi):
        roi_id = self.__roi_2_id(roi)
        roi_img = full_img[roi.rect.tl.row:roi.rect.br.row,
                  roi.rect.tl.col:roi.rect.br.col]

        height = self.predictor_dict[mq_param.ROIS].height
        width = self.predictor_dict[mq_param.ROIS].width
        roi_img, _, _ = image_api.resize_image_fill(roi_img, height, width, 3)

        return roi_id, roi_img

    def __filter_rois_by_size(self, rois):
        min_component_size = self.comp_config[mq_param.ROIS][mq_param.MIN_COMPONENT_SIZE]
        return [r for r in rois if r.rect.br.row - r.rect.tl.row >= min_component_size]

    def __filter_rois_by_class(self, rois):
        return [r for r in rois if proto_api.get_roi_type_name(r.type) in self.target_roi_classes]

    def __get_roi_preprocess_data(self, image_proto, img):

        id_2_rois = {}

        rois = self.__filter_rois_by_class(image_proto.rois)
        rois = self.__filter_rois_by_size(rois)

        for roi in rois:
            roi_id, roi_img = self.__extract_cropped_roi(img, roi)
            id_2_rois[roi_id] = roi_img

        return id_2_rois

    def __make_prediction_on_img(self, id_2_components, ocr_predictor):
        ids, imgs = id_2_components.keys(), id_2_components.values()

        pred_2_confs = [ocr_predictor.make_prediction_on_img(img) for img in imgs]
        pred_2_confs = [(ocr_predictor.postprocess_text(pred), conf) for pred, conf in pred_2_confs]
        ids_2_predictions = dict(zip(ids, pred_2_confs))

        return ids_2_predictions

    def preprocess(self, image_proto):
        img = image_api.get_rgb(image_proto.metadata.image_path)

        signpost_preprocess_data = self.__get_signpost_comp_preprocess_data(image_proto, img)
        rois_preprocess_data = self.__get_roi_preprocess_data(image_proto, img)

        preprocess_data = {
            mq_param.SIGNPOST_COMP: signpost_preprocess_data,
            mq_param.ROIS: rois_preprocess_data
        }
        return preprocess_data

    def predict(self, preprocess_data_list):

        predict_data_list = []

        for preprocess_data in preprocess_data_list:
            signpost_comp_preprocess_data = preprocess_data[mq_param.SIGNPOST_COMP]
            rois_preprocess_data = preprocess_data[mq_param.ROIS]

            signpost_comp_preds = self.__make_prediction_on_img(signpost_comp_preprocess_data,
                                                                self.predictor_dict[mq_param.SIGNPOST_COMP])
            roi_preds = self.__make_prediction_on_img(rois_preprocess_data,
                                                      self.predictor_dict[mq_param.ROIS])

            predict_data = {
                mq_param.SIGNPOST_COMP: signpost_comp_preds,
                mq_param.ROIS: roi_preds
            }

            predict_data_list.append(predict_data)

        return predict_data_list

    def postprocess(self, predict_data, image_proto):

        # todo: filter by conf thresh

        signpost_comp_preds = predict_data[mq_param.SIGNPOST_COMP]
        roi_preds = predict_data[mq_param.ROIS]

        img = image_api.get_rgb(image_proto.metadata.image_path)

        for signpost_comp_id, (text_pred,text_conf) in signpost_comp_preds.items():
            tl_row = int(signpost_comp_id.split("-")[0])
            tl_col = int(signpost_comp_id.split("-")[1])
            br_row = int(signpost_comp_id.split("-")[2])
            br_col = int(signpost_comp_id.split("-")[3])

            signpost_img = img[tl_row:br_row,tl_col:br_col]
            print(text_pred,text_conf)
            plt.imshow(signpost_img)
            plt.show()

        for roi_id, (text_pred, text_conf) in roi_preds.items():
            tl_row = int(roi_id.split("-")[0])
            tl_col = int(roi_id.split("-")[1])
            br_row = int(roi_id.split("-")[2])
            br_col = int(roi_id.split("-")[3])

            roi_img = img[tl_row:br_row, tl_col:br_col]
            print(text_pred, text_conf)
            plt.imshow(roi_img)
            plt.show()

        return image_proto


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", help="config file path", type=str, required=True)

    return parser.parse_args()


def run_predictor(conf_file):
    conf = io_utils.config_load(conf_file)
    predictor = OCR_MQ_Predictor(conf)
    predictor.start()


if __name__ == '__main__':
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    args = parse_arguments()
    run_predictor(args.config_file)
