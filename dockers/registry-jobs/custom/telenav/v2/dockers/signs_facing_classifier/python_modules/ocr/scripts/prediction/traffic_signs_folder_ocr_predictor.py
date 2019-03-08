import os
import logging
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from glob import glob
import editdistance
from tqdm import tqdm

from ocr.scripts.prediction.ocr_predictor import OCRPredArgs as ocr_args
from ocr.scripts.prediction.folder_ocr_predictor import FolderOCRPredictor
from ocr.scripts.text_correction.traffic_signs_text_corrector import TrafficSignsTextCorrector

import apollo_python_common.log_util as log_util
import apollo_python_common.io_utils as io_utils
import apollo_python_common.ml_pipeline.config_api as config_api
import apollo_python_common.image as image_api
from ocr.scripts.prediction.ocr_predictor import ResizeType

def predict_on_folder(config):
    nr_imgs = config_api.get_config_param(ocr_args.NR_IMGS_CONFIG_KEY, config, None)
    with_gt = config_api.get_config_param(ocr_args.WITH_EVALUATE_CONFIG_KEY, config, False)
    with_text_correction = config_api.get_config_param(ocr_args.WITH_TEXT_CORRECTION, config, False)
    text_correction_resources_path = config_api.get_config_param(ocr_args.TEXT_CORRECTION_RESOURCES_PATH, config, "")
    min_component_size = config_api.get_config_param(ocr_args.MIN_COMPONENT_SIZE_CONFIG_KEY, config, 0)
    conf_thresh = config_api.get_config_param(ocr_args.CONF_THRESH_CONFIG_KEY, config, 0)
    
    if with_text_correction:
        text_corrector = TrafficSignsTextCorrector(text_correction_resources_path)
    else:
        text_corrector = None
        
    predictor = FolderOCRPredictor(config.dataset_name, config.ckpt_path, ResizeType.PAD, text_corrector)
    pred_df = predictor.predict_on_folder(config.predict_folder,
                                          nr_imgs=nr_imgs,
                                          with_gt=with_gt,
                                          min_component_size=min_component_size
                                          )
    pred_df = pred_df[pred_df[predictor.PRED_CONF] > conf_thresh]

    if with_gt:
        print("Percentage correct = {}".format(round(pred_df[predictor.CORRECT].mean(), 4)))

    if ocr_args.OUTPUT_PATH_CONFIG_KEY in config:
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
