import os
import PIL
import logging
import argparse
import pandas as pd
from multiprocessing import Pool
from functools import partial
import multiprocessing
import shutil

from ocr.scripts.fake_dataset.traffic_signs_text_generator import TrafficSignsTextGenerator
from ocr.scripts.fake_dataset.traffic_signs_image_generator import TrafficSignsImageGenerator
from ocr.scripts.fake_dataset.fake_dataset_generator import FakeDatasetGenerator
from ocr.scripts.fake_dataset.tf_record import TFRecordDatasetCreator, TFRecordGenerator

import apollo_python_common.log_util as log_util
import apollo_python_common.io_utils as io_utils

IMAGES_FOLDER = "images"
TF_DATA_FOLDER = "tf_data"


class TrafficSignsFakeDatasetGenerator(FakeDatasetGenerator):
    NEW_LINE_PROXY_CHAR = "€"
    NEW_LINE_CHAR = "\n"

    def preprocess_data(self, data):
        return [(img_name, text.replace(self.NEW_LINE_CHAR, self.NEW_LINE_PROXY_CHAR)) for img_name, text in data]


def generate_images(config, imgs_save_path):
    img_generator_config = dict(io_utils.json_load(config.img_generator_config_path))
    text_generator = TrafficSignsTextGenerator(config.resources_path, config.nr_texts)
    image_generator = TrafficSignsImageGenerator(img_generator_config)

    TrafficSignsFakeDatasetGenerator(imgs_save_path, image_generator, text_generator).create_dataset()


def generate_tf_records(config, imgs_save_path, tf_data_save_path):
    TFRecordGenerator(config).create_tf_records(imgs_save_path, tf_data_save_path)


def generate_fake_dataset(config):
    imgs_save_path = os.path.join(config.save_path, IMAGES_FOLDER)
    tf_data_save_path = os.path.join(config.save_path, TF_DATA_FOLDER)

    generate_images(config, imgs_save_path)
    generate_tf_records(config, imgs_save_path, tf_data_save_path)


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
        generate_fake_dataset(input_config)
    except Exception as err:
        logger.error(err, exc_info=True)
        raise err
