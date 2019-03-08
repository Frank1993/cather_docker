import sys
import argparse
import logging
import threading

import orbb_definitions_pb2 as orbb_definitions
import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
import apollo_python_common.image
import apollo_python_common.proto_api as proto_api
import object_detection.retinanet.utils as retina_utils
import object_detection.retinanet.predict as predict

from scripts.sign_elements_data import SignElementsLabels
from apollo_python_common.ml_pipeline.multi_threaded_predictor import MultiThreadedPredictor
from apollo_python_common.io_utils import json_load
from apollo_python_common.image import OscDetails


class SignpostComponentsPredictor(MultiThreadedPredictor):
    '''
    Multi threaded predictor using RetinaNet model
    '''

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.log_level = 0
        io_utils.require_paths([config.train_meta_file,
                                config.score_thresholds_file,
                                config.weights_file])
        self.rois_labels = SignElementsLabels(config.train_meta_file)
        self.score_threshold_per_class = json_load(config.score_thresholds_file)
        self.models_per_worker = dict()
        self.resolutions = config.predict_resolutions
        self.min_side_size = config.predict_min_side_size

    def __get_cropped_roi(self, roi, image):
        return image[roi.rect.tl.row: roi.rect.br.row, roi.rect.tl.col: roi.rect.br.col]

    def __get_model_for_thread(self):
        thread_id = threading.get_ident()
        if thread_id not in self.models_per_worker:
            self.models_per_worker[thread_id] = predict.get_model_for_pred(self.config)
        return self.models_per_worker[thread_id]

    def __add_components_to_proto(self, roi, detections):
        coords_list = detections[0][0]
        conf_list = detections[0][1]
        type_list = detections[0][2]

        coords_2_conf_2_type_list = list(zip(coords_list, conf_list, type_list))

        for coords, confidence, roi_type in coords_2_conf_2_type_list:
            component = roi.components.add()
            component.type = proto_api.get_component_type_value(roi_type)
            component.box.tl.row = int(coords[1]) + roi.rect.tl.row
            component.box.tl.col = int(coords[0]) + roi.rect.tl.col
            component.box.br.row = int(coords[3]) + roi.rect.tl.row
            component.box.br.col = int(coords[2]) + roi.rect.tl.col
            component.confidence = confidence

    def preprocess(self, image_proto):
        signpost_rois = [roi for roi in image_proto.rois if
                         roi.type == orbb_definitions.SIGNPOST_GENERIC]
        if not signpost_rois:
            return []

        osc_details = OscDetails(image_proto.metadata.id, self.config.osc_api_url)
        image = apollo_python_common.image.get_bgr(image_proto.metadata.image_path, osc_details)
        preproc_data = [(roi.rect, self.__get_cropped_roi(roi, image)) for roi in signpost_rois]

        return preproc_data

    def predict(self, list_of_crop_lists):
        model = self.__get_model_for_thread()
        predictions_list = []
        for crop_list in list_of_crop_lists:
            crop_predictions = []
            for rect, img in crop_list:
                predicts = retina_utils.predict_images_on_batch([img], model, self.rois_labels,
                                                                self.resolutions, self.score_threshold_per_class,
                                                                self.log_level)
                crop_predictions.append((rect, predicts))
            predictions_list.append(crop_predictions)
        return predictions_list

    def postprocess(self, one_file_predictions, image_proto):
        signpost_rois = [roi for roi in image_proto.rois if
                         roi.type == orbb_definitions.SIGNPOST_GENERIC]
        for roi in signpost_rois:
            for rect, detections in one_file_predictions:
                if rect == roi.rect:
                    self.__add_components_to_proto(roi, detections)
        return image_proto


def run_predictor(conf_file):
    conf = io_utils.config_load(conf_file)
    predictor = SignpostComponentsPredictor(conf)
    predictor.start()


def __parse_args(args):
    parser = argparse.ArgumentParser(description='Signs components detector. Takes images metadata from message queue.')
    parser.add_argument('--config_files',
                        help='Configuration file''s path',
                        default='../../../../signpost_components_detector_retinanet_config.json')
    return parser.parse_args(args)


if __name__ == '__main__':
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    # parse arguments
    args = sys.argv[1:]
    args = __parse_args(args)
    run_predictor(args.config_files)
