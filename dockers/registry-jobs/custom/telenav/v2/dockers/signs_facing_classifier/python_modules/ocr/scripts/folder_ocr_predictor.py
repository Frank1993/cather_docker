import os
import logging
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from glob import glob
import editdistance
from tqdm import tqdm

from ocr.scripts.ocr_predictor import OCRPredictor

import apollo_python_common.log_util as log_util
import apollo_python_common.io_utils as io_utils
import apollo_python_common.ml_pipeline.config_api as config_api
import apollo_python_common.image as image_api

MIN_COMPONENT_SIZE_CONFIG_KEY = "min_component_size"
NR_IMGS_CONFIG_KEY = "nr_imgs"
WITH_EVALUATE_CONFIG_KEY = "with_evaluate"
OUTPUT_PATH_CONFIG_KEY = "output_csv_path"
SPELL_CHECKER_RES_PATH_CONFIG_KEY = "spell_checker_resources_path"
CONF_THRESH_CONFIG_KEY = "conf_thresh"


class FolderOCRPredictor(OCRPredictor):
    IMG_PATH = "img_path"
    IMG = "img"
    PRED_TEXT = "pred_text"
    PRED_CONF = "pred_conf"
    GT_TEXT = "gt_text"
    WIDTH = "width"
    HEIGHT = "height"
    AREA = "area"
    ERROR = "error"
    ERROR_BUCKET = "error_bucket"
    HEIGHT_BUCKET = "height_bucket"
    WIDTH_BUCKET = "width_bucket"
    AREA_BUCKET = "area_bucket"
    CORRECT = "correct"
    ASPECT_RATIO = "aspect_ratio"
    GT_CHAR_LENGTH = "gt_char_length"
    GT_WORD_LENGTH = "gt_word_length"

    def __init__(self, dataset_name, ckpt_path, spell_checker_resources_path):
        super().__init__(dataset_name, ckpt_path, spell_checker_resources_path)
        tf.reset_default_graph()

    def __compute_pred_df(self, pred_df, pred_2_confs, with_gt):
        pred_df[self.PRED_TEXT] = pd.Series([p for p, c in pred_2_confs], index=pred_df.index)
        pred_df[self.PRED_TEXT] = pred_df[self.PRED_TEXT].progress_apply(self.postprocess_text)
        pred_df[self.PRED_CONF] = pd.Series([c for p, c in pred_2_confs], index=pred_df.index)

        if with_gt:
            pred_df[self.GT_TEXT] = pred_df[self.IMG_PATH].apply(lambda img_path: self.__get_gt_from_img_path(img_path))
            pred_df[self.ERROR] = pred_df.apply(lambda r: self.__compute_error(r[self.GT_TEXT], r[self.PRED_TEXT]),
                                                axis=1)
            pred_df[self.CORRECT] = pred_df[self.ERROR].apply(lambda er: 1 if er == 0 else 0)
            pred_df[self.GT_CHAR_LENGTH] = pred_df[self.GT_TEXT].apply(len)
            pred_df[self.GT_WORD_LENGTH] = pred_df[self.GT_TEXT].apply(lambda pred: len(pred.split(" ")))

        return pred_df

    def __compute_error(self, gt, pred):
        return editdistance.eval(gt, pred)

    def __get_gt_from_img_path(self, img_path):
        return os.path.basename(os.path.splitext(img_path)[0]).split("_")[-1].lower().replace(":", "/")

    def __get_data_df(self, img_paths, nr_imgs, min_component_size):
        data_df = pd.DataFrame({self.IMG_PATH: img_paths})
        data_df[self.IMG] = data_df[self.IMG_PATH].apply(image_api.get_rgb)
        data_df[self.WIDTH] = data_df[self.IMG].apply(lambda img: img.shape[1])
        data_df[self.HEIGHT] = data_df[self.IMG].apply(lambda img: img.shape[0])

        data_df[self.AREA] = data_df[self.WIDTH] * data_df[self.HEIGHT]
        data_df[self.ASPECT_RATIO] = data_df.apply(lambda row: row[self.WIDTH] // row[self.HEIGHT], axis=1)

        data_df = data_df[data_df[self.HEIGHT] > min_component_size]

        data_df = data_df[:nr_imgs]

        return data_df

    def __get_imgs(self, data_df):
        imgs = [image_api.cv_resize(img, self.width, self.height, default_interpolation=False) for img in
                data_df[self.IMG].values]
        imgs = np.stack(imgs)
        return imgs

    def predict_on_folder(self, predict_folder, nr_imgs=None, with_gt=False, min_component_size=0):
        img_paths = glob(predict_folder + "/*")
        nr_imgs = len(img_paths) if nr_imgs is None else nr_imgs
        data_df = self.__get_data_df(img_paths, nr_imgs, min_component_size)
        imgs = self.__get_imgs(data_df)

        pred_2_conf = [self.make_prediction_on_img(img) for img in tqdm(imgs)]
        pred_df = self.__compute_pred_df(data_df, pred_2_conf, with_gt)

        return pred_df


def predict_on_folder(config):
    nr_imgs = config_api.get_config_param(NR_IMGS_CONFIG_KEY, config, None)
    with_gt = config_api.get_config_param(WITH_EVALUATE_CONFIG_KEY, config, False)
    spell_checker_resources_path = config_api.get_config_param(SPELL_CHECKER_RES_PATH_CONFIG_KEY, config, None)
    min_component_size = config_api.get_config_param(MIN_COMPONENT_SIZE_CONFIG_KEY, config, 0)
    conf_thresh = config_api.get_config_param(CONF_THRESH_CONFIG_KEY, config, 0)

    predictor = FolderOCRPredictor(config.dataset_name, config.ckpt_path, spell_checker_resources_path)
    pred_df = predictor.predict_on_folder(config.predict_folder,
                                          nr_imgs=nr_imgs,
                                          with_gt=with_gt,
                                          min_component_size=min_component_size
                                          )
    pred_df = pred_df[pred_df[predictor.PRED_CONF] > conf_thresh]

    if with_gt:
        print("Percentage correct = {}".format(round(pred_df[predictor.CORRECT].mean(), 4)))

    if OUTPUT_PATH_CONFIG_KEY in config:
        pred_df[[predictor.IMG_PATH, predictor.PRED_TEXT]].to_csv(config.output_csv_path, index=False)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--config_json", help="path to config json", type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':

    log_util.config(__file__)
    logger = logging.getLogger(__name__)

    args = parse_arguments()
    input_config = io_utils.json_load(args.config_json)

    try:
        predict_on_folder(input_config)
    except Exception as err:
        logger.error(err, exc_info=True)
        raise err
