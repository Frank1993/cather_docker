import sys
import argparse
import logging
import threading
import apollo_python_common.image
from apollo_python_common.ml_pipeline.multi_threaded_predictor import MultiThreadedPredictor
import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
from apollo_python_common.io_utils import json_load
from scripts.rois_data import RoisTypes
import object_detection.retinanet.utils as retina_utils
import object_detection.retinanet.predict as predict
import apollo_python_common.proto_api as proto_api
from vanishing_point.vanishing_point import VanishingPointDetector


class RetinaNetPredictor(MultiThreadedPredictor):
    '''
    Multi threaded predictor using RetinaNet model
    '''
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.log_level = 0
        io_utils.require_paths([config.train_meta_file,
                                config.score_thresholds_file,
                                config.weights_file])
        self.rois_labels = RoisTypes(config.train_meta_file)
        self.score_threshold_per_class = json_load(config.score_thresholds_file)
        self.models_per_worker = dict()
        self.resolutions = config.predict_resolutions
        self.min_side_size = config.predict_min_side_size
        self.cut_below_vanishing_point = config.cut_below_vanishing_point
        self.vp_detector_per_worker = dict()

    def preprocess(self, image_proto):
        if threading.get_ident() not in self.vp_detector_per_worker:
            self.vp_detector_per_worker[threading.get_ident()] = VanishingPointDetector()
        osc_details = apollo_python_common.image.OscDetails(image_proto.metadata.id, self.config.osc_api_url)
        image = apollo_python_common.image.get_bgr(image_proto.metadata.image_path, osc_details)

        _, preprocessed_image = retina_utils.preprocess_image(image, self.cut_below_vanishing_point)
        proto_api.add_image_size(image_proto, image.shape)
        proto_api.add_vanishing_point(image_proto, image, self.vp_detector_per_worker[threading.get_ident()])
        return preprocessed_image

    def predict(self, input_msg_list):
        if threading.get_ident() not in self.models_per_worker:
            self.models_per_worker[threading.get_ident()] = predict.get_model_for_pred(self.config)
        model = self.models_per_worker[threading.get_ident()]
        predictions_list = retina_utils.predict_images_on_batch(input_msg_list, model, self.rois_labels,
                            self.resolutions, self.score_threshold_per_class, self.log_level)
        return predictions_list

    def postprocess(self, one_file_predictions, image_proto):
        (boxes, scores, label_names) = one_file_predictions
        retina_utils.add_detections_to_img_proto(self.config.algorithm, self.config.algorithm_version, image_proto,
                                                 boxes, scores, label_names, self.min_side_size)
        return image_proto


def run_predictor(conf_file):
    conf = io_utils.config_load(conf_file)
    predictor = RetinaNetPredictor(conf)
    predictor.start()


def __parse_args(args):
    parser = argparse.ArgumentParser(description='Traffic signs detector. Takes images metadata from message queue.')
    parser.add_argument('--config_files',
                        help='Configuration file''s path',
                        default='../../../../traffic_signs_detector_retinanet_config.json')
    return parser.parse_args(args)


if __name__ == '__main__':
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    # parse arguments
    args = sys.argv[1:]
    args = __parse_args(args)
    run_predictor(args.config_files)
