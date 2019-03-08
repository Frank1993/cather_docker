import os

import apollo_python_common.image as image
import apollo_python_common.ml_pipeline.config_api as config_api
import apollo_python_common.proto_api as proto_api
import classification.scripts.constants as constants
import classification.scripts.dataset_builder as builder
import classification.scripts.network as network
import classification.scripts.utils as utils
import keras
import numpy as np
import tensorflow as tf
from apollo_python_common.ml_pipeline.config_api import MQ_Param
from apollo_python_common.ml_pipeline.multi_threaded_predictor import MultiThreadedPredictor
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inc_v3


class AbstractClassifPredictor(MultiThreadedPredictor):

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.model = None
        self.index_2_algorithm_name, self.index_2_params = self.__compute_params_dicts()


    def __set_tf_session(self):

        gpu_fraction_param = config_api.get_config_param(MQ_Param.PER_PROCESS_GPU_MEMORY_FRACTION, self.config,
                                                         default_value=None)
        allow_growth_param = config_api.get_config_param(MQ_Param.ALLOW_GROWTH_GPU_MEMORY, self.config,
                                                         default_value=None)

        if gpu_fraction_param is not None and allow_growth_param is not None:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction_param,
                                        allow_growth=allow_growth_param)
        else:
            gpu_options = tf.GPUOptions()

        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        keras.backend.tensorflow_backend.set_session(sess)

    def __is_alg_active(self, alg_name):
        return "{}_bundle_path".format(alg_name) in self.config

    def get_alg_bundle_path(self, alg_name):
        return config_api.get_config_param("{}_bundle_path".format(alg_name), self.config, default_value="")

    def __compute_params_dicts(self):
        active_algorithms = [alg_name for alg_name in constants.AVAILABLE_ALGORITHMS if self.__is_alg_active(alg_name)]

        i2a_list = list(enumerate(active_algorithms))

        index_2_algorithm_name = {i: alg_name for i, alg_name in i2a_list}
        index_2_params = {i: self.__get_training_params(alg_name) for i, alg_name in i2a_list}

        return index_2_algorithm_name, index_2_params

    def __get_training_params(self, alg_name):
        path = os.path.join(self.get_alg_bundle_path(alg_name), "model_params.json")
        return utils.json_load_classif_params(path)

    def __get_trained_model(self, alg_name):
        bundle_path = self.get_alg_bundle_path(alg_name)

        model_structure_path = os.path.join(bundle_path, "model_structure.json")
        model_weights_path = os.path.join(bundle_path, "model_weights.h5")

        return network.load_model_from_json_and_weights(model_structure_path, model_weights_path)

    def __compute_model_dict(self):
        return {index: self.__get_trained_model(alg_name) for index, alg_name in
                self.index_2_algorithm_name.items()}

    def __generic_params(self):
        return self.index_2_params[0]

    def resize(self, img):
        keep_aspect = self.__generic_params().keep_aspect
        new_img_size = self.__generic_params().img_size

        if keep_aspect:
            img, _, _ = image.resize_image_fill(img, new_img_size[1], new_img_size[0], 3)
        else:
            img = image.cv_resize(img, new_img_size[0], new_img_size[1])

        return img.astype(np.float32)

    def preprocess_image_according_to_backbone(self, img):
        return preprocess_input_inc_v3(img)

    def load_model(self):
        self.__set_tf_session()
        index_2_model = self.__compute_model_dict()
        return network.get_hydra_model(self.__generic_params(), index_2_model, self.index_2_algorithm_name)

    def nr_prediction_heads(self):
        return len(self.index_2_algorithm_name)

    def predict_with_model(self, imgs):

        imgs = np.stack(imgs)
        predictions = self.model.predict(imgs)

        if self.nr_prediction_heads() != 1:
            predictions = np.asarray(list(zip(*predictions)))

        return predictions

    def read_image(self, image_proto):
        img = image.get_rgb(image_proto.metadata.image_path)
        proto_api.add_image_size(image_proto, img.shape)

        if self.__generic_params().with_vp_crop:
            img = builder.crop_at_horizon_line(img)

        return img

    def preprocess(self, image_proto):
        raise NotImplementedError('Method not implemented')

    def predict(self, images):
        raise NotImplementedError('Method not implemented')

    def postprocess(self, predictions, image_proto):
        raise NotImplementedError('Method not implemented')
