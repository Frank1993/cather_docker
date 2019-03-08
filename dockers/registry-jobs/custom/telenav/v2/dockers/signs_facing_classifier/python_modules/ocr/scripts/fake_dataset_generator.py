import os
import PIL
import logging
import argparse
import pandas as pd
from multiprocessing import Pool
from functools import partial
import multiprocessing
import shutil

from ocr.scripts.text_generator import TextGenerator
from ocr.scripts.image_generator import ImageGenerator
from ocr.scripts.tf_record_dataset_creator import TFRecordDatasetCreator

import apollo_python_common.log_util as log_util
import apollo_python_common.io_utils as io_utils


TRAIN_FOLDER = "train"
TEST_FOLDER = "test"
IMAGES_FOLDER = "images"
TF_DATA_FOLDER = "tf_data"
CHARSET_FILE_NAME = "charset_size=134.txt"
TRAIN_PATTERN = "train-"
TEST_PATTERN = "test-"

class FakeDatasetGenerator:
    TRAIN_PERCENTAGE = 0.99
    IMG_NAME_COL = "img_name"
    TEXT_COL = "text"
    OUTPUT_CSV_NAME = "ocr_data.csv"

    def __init__(self, resources_path, save_path, img_width, img_height, nr_texts_per_length, char_limit=16):
        self.train_save_path = os.path.join(save_path, TRAIN_FOLDER)
        self.test_save_path = os.path.join(save_path, TEST_FOLDER)
        self.image_generator = ImageGenerator(img_width, img_height, resources_path)
        self.text_generator = TextGenerator(resources_path)
        self.nr_texts_per_length = nr_texts_per_length
        self.char_limit = char_limit

    def __split_train_test(self, data, train_percentage):
        nr_train = int(train_percentage * len(data))
        train_data = data[:nr_train]
        test_data = data[nr_train:]

        return train_data, test_data

    def generate_img_and_write_to_disk(self, base_path, img_name_2_text):
        img_name, text = img_name_2_text
        os.makedirs(base_path, exist_ok=True)
        img = self.image_generator.generate_img(text)

        PIL.Image.fromarray(img).save(os.path.join(base_path, img_name))

    def __save_csv(self,data,path):
        io_utils.create_folder(path)
        pd.DataFrame(data,columns=[self.IMG_NAME_COL, self.TEXT_COL])\
                    .to_csv(os.path.join(path,self.OUTPUT_CSV_NAME),index=False)
        
    def create_dataset(self):
        texts = self.text_generator.generate_texts(self.nr_texts_per_length, self.char_limit)
        img_names_2_texts = [(f"img_{i}.jpg",text) for i,text in enumerate(texts)]

        train_data, test_data = self.__split_train_test(img_names_2_texts, self.TRAIN_PERCENTAGE)
        
        self.__save_csv(train_data,self.train_save_path)
        self.__save_csv(test_data,self.test_save_path)
        
        pool = Pool(multiprocessing.cpu_count() // 2)
        pool.map(partial(self.generate_img_and_write_to_disk, self.train_save_path), train_data)
        pool.close()

        pool = Pool(multiprocessing.cpu_count() // 2)
        pool.map(partial(self.generate_img_and_write_to_disk, self.test_save_path), test_data)
        pool.close()


def generate_images(config, imgs_save_path):
    FakeDatasetGenerator(config.resources_path,
                         imgs_save_path,
                         config.width, config.height,
                         config.nr_texts_per_length, config.char_limit).create_dataset()
    
def generate_tf_records(config, imgs_save_path, tf_data_save_path):
    
    old_charset_path = os.path.join(config.resources_path,CHARSET_FILE_NAME)
    new_charset_path = os.path.join(tf_data_save_path,CHARSET_FILE_NAME)

    io_utils.create_folder(tf_data_save_path)
    shutil.copyfile(old_charset_path, new_charset_path)
    
    train_input_path = os.path.join(imgs_save_path, TRAIN_FOLDER)
    train_output_path = os.path.join(tf_data_save_path, TRAIN_FOLDER)

    test_input_path = os.path.join(imgs_save_path, TEST_FOLDER)
    test_output_path = os.path.join(tf_data_save_path, TEST_FOLDER)
    
    TFRecordDatasetCreator(train_input_path, train_output_path, new_charset_path, TRAIN_PATTERN).create_dataset()
    TFRecordDatasetCreator(test_input_path, test_output_path, new_charset_path, TEST_PATTERN).create_dataset()
    
    
def generate_fake_dataset(config):
    imgs_save_path = os.path.join(config.save_path,IMAGES_FOLDER)
    tf_data_save_path = os.path.join(config.save_path,TF_DATA_FOLDER)
                    
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
