import argparse
import logging

from tqdm import tqdm

tqdm.pandas()

import apollo_python_common.image as image_api
import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
import apollo_python_common.proto_api as proto_api
import apollo_python_common.ml_pipeline.config_api as config_api
from apollo_python_common.ml_pipeline.multi_threaded_predictor import MultiThreadedPredictor
from apollo_python_common.ml_pipeline.config_api import MQ_Param
from ocr.scripts.prediction.ocr_predictor import OCRPredictor
from ocr.scripts.text_correction.signpost_text_corrector import SignpostTextCorrector


class OCR_MQ_Param(MQ_Param):
    DATASET_CONFIG_KEY = "dataset"
    CKPT_PATH_CONFIG_KEY = "ckpt_path"
    SPELL_CHECKER_PATH_CONFIG_KEY = "spell_checker_resources_path"
    CUSTOM_CHARSET_PATH = "custom_charset_path"
    MIN_COMPONENT_SIZE_CONFIG_KEY = "min_component_size"
    CONF_THRESH_CONFIG_KEY = "conf_thresh"


class OCR_MQ_Predictor(MultiThreadedPredictor):
    SIGNPOST_GENERIC_PROTO_NAME = "SIGNPOST_GENERIC"
    GENERIC_TEXT_COMPONENT_NAME = "GENERIC_TEXT"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.ocr_predictor = self.__build_ocr_predictor()
        self.min_component_size = config_api.get_config_param(OCR_MQ_Param.MIN_COMPONENT_SIZE_CONFIG_KEY, config, 25)
        self.conf_thresh = config_api.get_config_param(OCR_MQ_Param.CONF_THRESH_CONFIG_KEY, config, 0)

    def __build_ocr_predictor(self):
        spell_checker_resources_path = config_api.get_config_param(OCR_MQ_Param.SPELL_CHECKER_PATH_CONFIG_KEY,
                                                                   self.config, None)
        custom_charset_path = config_api.get_config_param(OCR_MQ_Param.CUSTOM_CHARSET_PATH, self.config, None)

        return OCRPredictor(config_api.get_config_param(OCR_MQ_Param.DATASET_CONFIG_KEY, self.config, ""),
                            config_api.get_config_param(OCR_MQ_Param.CKPT_PATH_CONFIG_KEY, self.config, ""),
                            SignpostTextCorrector(spell_checker_resources_path),
                            custom_charset_path = custom_charset_path
                           )

    def __comp_2_id(self, comp_proto):
        return "{}-{}-{}-{}".format(comp_proto.box.tl.row,
                                    comp_proto.box.tl.col,
                                    comp_proto.box.br.row,
                                    comp_proto.box.br.col)

    def __extract_cropped_comp(self, full_img, comp_proto):
        comp_id = self.__comp_2_id(comp_proto)
        comp_img = full_img[comp_proto.box.tl.row:comp_proto.box.br.row,
                   comp_proto.box.tl.col:comp_proto.box.br.col]

        comp_img = image_api.cv_resize(comp_img, self.ocr_predictor.width, self.ocr_predictor.height)
        return comp_id, comp_img

    def __get_signpost_rois(self, image_proto):
        return [roi for roi in image_proto.rois if
                proto_api.get_roi_type_name(roi.type) == self.SIGNPOST_GENERIC_PROTO_NAME]

    def __get_text_components(self, roi):
        return [c for c in roi.components if
                proto_api.get_component_type_name(c.type) == self.GENERIC_TEXT_COMPONENT_NAME]

    def __filter_components_by_size(self, text_components):
        return [c for c in text_components if c.box.br.row - c.box.tl.row >= self.min_component_size]

    def preprocess(self, image_proto):
        signpost_rois = self.__get_signpost_rois(image_proto)
        
        img = None 
        id_2_components = {}
        
        for roi in signpost_rois:
            text_components = self.__get_text_components(roi)
            text_components = self.__filter_components_by_size(text_components)
            for comp_proto in text_components:
                img = image_api.get_rgb(image_proto.metadata.image_path) if img is None else img
                comp_id, comp_img = self.__extract_cropped_comp(img, comp_proto)
                id_2_components[comp_id] = comp_img

        return id_2_components

    def predict(self, id_2_components_list):

        ids_2_predictions_list = []
        for id_2_components in id_2_components_list:
            ids, imgs = id_2_components.keys(), id_2_components.values()
            pred_2_confs = [self.ocr_predictor.make_prediction_on_img(img) for img in imgs]
            ids_2_predictions = dict(zip(ids, pred_2_confs))
            ids_2_predictions_list.append(ids_2_predictions)

        return ids_2_predictions_list

    def postprocess(self, ids_2_predictions, image_proto):

        signpost_rois = self.__get_signpost_rois(image_proto)
        for roi in signpost_rois:
            text_components = self.__get_text_components(roi)
            for comp_proto in text_components:
                comp_id = self.__comp_2_id(comp_proto)
                if comp_id in ids_2_predictions:
                    text, conf = ids_2_predictions[comp_id]
                    if conf > self.conf_thresh:
                        text = self.ocr_predictor.postprocess_text(text)
                        comp_proto.value = text

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
