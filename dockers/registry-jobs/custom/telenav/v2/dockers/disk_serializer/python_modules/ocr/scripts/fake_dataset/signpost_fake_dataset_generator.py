import argparse
import logging
import os

import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
from ocr.scripts.fake_dataset.fake_dataset_generator import FakeDatasetGenerator
from ocr.scripts.fake_dataset.poi_text_generator import PoiTextGenerator
from ocr.scripts.fake_dataset.signpost_image_generator import SignpostImageGenerator
from ocr.scripts.fake_dataset.tf_record import TFRecordGenerator

IMAGES_FOLDER = "images"
TF_DATA_FOLDER = "tf_data"


def generate_images(config, imgs_save_path):
    img_generator_config = dict(io_utils.json_load(config.img_generator_config_path))

    text_generator = PoiTextGenerator(config.resources_path, config.nr_texts_per_length, config.char_limit)
    image_generator = SignpostImageGenerator(img_generator_config)

    FakeDatasetGenerator(imgs_save_path, image_generator, text_generator).create_dataset()


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
