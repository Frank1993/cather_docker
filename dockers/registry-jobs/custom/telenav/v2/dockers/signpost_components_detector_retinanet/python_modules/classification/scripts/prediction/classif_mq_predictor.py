import argparse
import logging

import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
import apollo_python_common.ml_pipeline.config_api as config_api
import numpy as np
from apollo_python_common.ml_pipeline.config_api import MQ_Param
from apollo_python_common.protobuf.classif_definitions_pb2 import *
from classification.scripts.prediction.abstract_classif_mq_predictor import AbstractClassifPredictor


class ClassifPredictor(AbstractClassifPredictor):

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def preprocess(self, image_proto):
        img = self.read_image(image_proto)
        img = self.resize(img)
        img = self.preprocess_image_according_to_backbone(img)

        return img

    def predict(self, images):

        if self.model is None:
            self.model = self.load_model()

        predictions = self.predict_with_model(images)

        return predictions

    def __exists_threshold_for_pred(self, alg_name, class_name, pred_thresholds_dict):
        return alg_name in pred_thresholds_dict and class_name in pred_thresholds_dict[alg_name]

    def __get_chosen_pred_class_name(self, valid_threshold_predictions):
        return sorted(valid_threshold_predictions, key=lambda class_2_conf: -class_2_conf[1])[0][0]

    def __adjust_predictions_format(self, predictions):
        if self.nr_prediction_heads() == 1:
            predictions = np.expand_dims(predictions, axis=0)

        return predictions

    def postprocess(self, predictions, image_proto):

        pred_thresholds_dict = config_api.get_config_param(MQ_Param.PRED_THRESHOLDS, self.config, default_value={})
        alg_version = config_api.get_config_param(MQ_Param.ALGORITHM_VERSION, self.config, default_value=-1)
        predictions = self.__adjust_predictions_format(predictions)

        for classif_index, alg_name in self.index_2_algorithm_name.items():
            classif_predictions = predictions[classif_index]
            classif_prediction_proto = image_proto.features.classif_predictions.add()

            valid_threshold_predictions = []
            for index, confidence in enumerate(list(classif_predictions)):
                classif_prediction_proto_class = classif_prediction_proto.pred_classes.add()
                classif_prediction_proto_class.class_name = self.index_2_params[classif_index].classIndex_2_class[index]
                classif_prediction_proto_class.confidence = round(float(confidence), 4)

                threshold_exists = self.__exists_threshold_for_pred(alg_name, classif_prediction_proto_class.class_name,
                                                                    pred_thresholds_dict)

                if threshold_exists:
                    threshold = pred_thresholds_dict[alg_name][classif_prediction_proto_class.class_name]
                    if classif_prediction_proto_class.confidence >= threshold:
                        valid_threshold_predictions.append((classif_prediction_proto_class.class_name,
                                                            classif_prediction_proto_class.confidence))
                else:
                    valid_threshold_predictions.append((classif_prediction_proto_class.class_name,
                                                        classif_prediction_proto_class.confidence))

            chosen_pred_class_name = self.__get_chosen_pred_class_name(valid_threshold_predictions)

            classif_prediction_proto.algorithm = alg_name
            classif_prediction_proto.algorithm_version = alg_version
            classif_prediction_proto.chosen_pred_class_name = chosen_pred_class_name

        return image_proto


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", help="config file path", type=str, required=True)

    return parser.parse_args()


def run_predictor(conf_file):
    conf = io_utils.config_load(conf_file)
    predictor = ClassifPredictor(conf)
    predictor.start()


if __name__ == '__main__':
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    args = parse_arguments()
    run_predictor(args.config_file)
