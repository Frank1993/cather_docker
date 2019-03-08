import os

import numpy as np
from tqdm import tqdm

tqdm.pandas()
import tensorflow as tf
from tensorflow.python.training import monitored_session

import common_flags
import datasets

common_flags.define()
import data_provider


class ResizeType:
    DEFORM = "deform"
    PAD = "pad"


class OCRPredArgs:
    MIN_COMPONENT_SIZE_CONFIG_KEY = "min_component_size"
    NR_IMGS_CONFIG_KEY = "nr_imgs"
    WITH_EVALUATE_CONFIG_KEY = "with_evaluate"
    WITH_TEXT_CORRECTION = "with_text_correction"
    WITH_SPELL_CHECK_CORRECTION = "with_spell_check_correction"
    OUTPUT_PATH_CONFIG_KEY = "output_csv_path"
    TEXT_CORRECTION_RESOURCES_PATH = "text_correction_resources_path"
    SPELL_CHECKER_RES_PATH_CONFIG_KEY = "spell_checker_resources_path"
    CONF_THRESH_CONFIG_KEY = "conf_thresh"


class OCRPredictor:
    IMAGE_SHAPE_KEY = "image_shape"
    NULL_CODE_KEY = "null_code"
    CHARSET_FILENAME_KEY = "charset_filename"

    def __init__(self, dataset_name, ckpt_path, text_corrector):
        self.width, self.height = self.__get_dataset_image_size(dataset_name)
        self.null_char = self.__get_null_char(dataset_name)
        self.text_corrector = text_corrector

        self.imgs_placeholder, self.endpoints = self.__get_model(dataset_name)
        self.graph = tf.get_default_graph()
        self.sess = self.__get_session(ckpt_path)

    def __get_session(self, ckpt_path):
        session_creator = monitored_session.ChiefSessionCreator(checkpoint_filename_with_path=ckpt_path)
        return monitored_session.MonitoredSession(session_creator=session_creator)

    def __create_dataset(self, dataset_name, split_name):
        ds_module = getattr(datasets, dataset_name)
        return ds_module.get_split(split_name)

    def __load_model(self, dataset_name):
        dataset = self.__create_dataset(dataset_name, split_name="train")
        model = common_flags.create_model(
            num_char_classes=dataset.num_char_classes,
            seq_length=dataset.max_sequence_length,
            num_views=dataset.num_of_views,
            null_code=dataset.null_code,
            charset=dataset.charset)

        return model

    def __get_null_char(self, dataset_name):
        ds_module = getattr(datasets, dataset_name)
        null_code = ds_module.DEFAULT_CONFIG[self.NULL_CODE_KEY]
        charset_name = ds_module.DEFAULT_CONFIG[self.CHARSET_FILENAME_KEY]
        charset = ds_module.read_charset(os.path.join(ds_module.DEFAULT_DATASET_DIR, charset_name))
        return charset[null_code]

    def __get_dataset_image_size(self, dataset_name):
        ds_module = getattr(datasets, dataset_name)
        height, width, _ = ds_module.DEFAULT_CONFIG[self.IMAGE_SHAPE_KEY]
        return width, height

    def postprocess_text(self, text):
        try:
            text = text.strip().decode("utf-8")
            if self.null_char in text:
                text = text[:text.index(self.null_char)]

            if self.text_corrector is not None:
                text = self.text_corrector.correct_text(text)

        except Exception as e:
            print(e)
            return ""

        return text

    def __get_model(self, dataset_name):
        model = self.__load_model(dataset_name)
        raw_images = tf.placeholder(tf.uint8, shape=[1, self.height, self.width, 3])
        images = tf.map_fn(data_provider.preprocess_image, raw_images, dtype=tf.float32)
        endpoints = model.create_base(images, labels_one_hot=None)
        return raw_images, endpoints

    def make_prediction_on_img(self, img):

        with self.graph.as_default():
            pred = self.sess.run(self.endpoints,
                                 feed_dict={self.imgs_placeholder: np.expand_dims(img, axis=0)})

            text = pred.predicted_text[0]
            confidence = np.mean(pred.predicted_scores)

        return text, confidence
