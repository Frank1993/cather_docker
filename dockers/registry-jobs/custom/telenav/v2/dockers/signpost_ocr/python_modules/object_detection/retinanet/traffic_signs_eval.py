import keras
import os
import logging
import apollo_python_common.proto_api as proto_api
from apollo_python_common.obj_detection_evaluator.model_statistics import ModelStatistics
from apollo_python_common.obj_detection_evaluator.protobuf_evaluator import convert_rois_dictionary
from object_detection.retinanet.utils import predict_folder
from scripts.traffic_signs_types import TrafficSignsTypes
from scripts.sign_components_types import SignComponentsTypes


class TrafficSignsEval(keras.callbacks.Callback):
    def __init__(self, generator, ground_truth_proto_file, train_proto_file,
                 resolution=300,
                 max_number_of_images=500,
                 roi_min_side_size=3,
                 lowest_score_threshold=0.5,
                 is_for_signpost_components=False):
        self.generator = generator
        self.ground_truth_proto_file = ground_truth_proto_file
        self.is_for_signpost_components = is_for_signpost_components
        if is_for_signpost_components:
            self.rois_labels = SignComponentsTypes(train_proto_file)
        else:
            self.rois_labels = TrafficSignsTypes(train_proto_file)
        self.resolution = resolution
        self.lowest_score_threshold = lowest_score_threshold
        self.images_folder = os.path.dirname(os.path.abspath(ground_truth_proto_file))
        self.max_number_of_images = max_number_of_images
        self.logger = logging.getLogger(__name__)
        self.ground_truth_metadata = proto_api.read_imageset_file(self.ground_truth_proto_file)
        self.ground_truth_proto_dictionary = proto_api.create_images_dictionary(self.ground_truth_metadata, True)
        self.roi_min_side_size = roi_min_side_size
        self.logger = logging.getLogger(__name__)
        super().__init__()

    def on_epoch_end(self, epoch, logs={}):
        self.logger.info("Evaluating checkpoint:")
        self.__evaluate_traffic_signs()

    def __evaluate_traffic_signs(self):
        resolutions = [self.resolution]
        score_threshold_per_class = dict([(class_label, self.lowest_score_threshold) for class_label in self.rois_labels.captions()])
        pred_metadata = predict_folder(self.model, self.images_folder, None, resolutions, self.rois_labels,
                                         score_threshold_per_class,
                                         draw_predictions=False, max_number_of_images=self.max_number_of_images,
                                         log_level=0)
        pred_proto_dictionary = proto_api.create_images_dictionary(pred_metadata, True)
        for file_name in list(self.ground_truth_proto_dictionary.keys()):
            if file_name not in pred_proto_dictionary:
                del self.ground_truth_proto_dictionary[file_name]

        ground_truth_detection_dictionary = convert_rois_dictionary(self.ground_truth_proto_dictionary,
                                                                    self.is_for_signpost_components)
        pred_detection_dictionary = convert_rois_dictionary(pred_proto_dictionary,
                                                            self.is_for_signpost_components)

        model_statistics = ModelStatistics(ground_truth_detection_dictionary, pred_detection_dictionary,
                                           self.roi_min_side_size)
        model_statistics.compute_model_statistics()

        self.logger.info(model_statistics.statistics['Total'])



