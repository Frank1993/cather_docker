import cv2
from copy import deepcopy

import orbb_definitions_pb2
import apollo_python_common.image as apollo_image
import apollo_python_common.proto_api as proto_api
import model.yolo_network_definition as network_definition
import tools.vp_utils as vp_utils
import apollo_python_common.io_utils as io_utils
from vanishing_point.vanishing_point import VanishingPointDetector

class ImageData:

    def __init__(self, image, image_width, image_height, width_crop_size,
                 height_crop_size, scaled_width, scaled_height, detections):
        self.image = image
        self.image_width = image_width
        self.image_height = image_height
        self.width_crop_size = width_crop_size
        self.height_crop_size = height_crop_size
        self.scaled_width = scaled_width
        self.scaled_height = scaled_height
        self.detections = detections


class YoloNetwork:

    CONFIDENCE_THRESH = 0.1
    HIERARCHICAL_THRESH = .5
    NMS_THRESH = .25

    def __init__(self, weights_path, yolo_network_config, yolo_metadata, yolo_classes_names):
        self.net_main, self.meta_main, self.alt_names = network_definition.load_network(yolo_network_config, weights_path,
                                                                                        yolo_metadata, yolo_classes_names)

    def detect(self, image):
        return network_definition.detect(self.net_main, self.meta_main, self.alt_names, image,
                                         self.CONFIDENCE_THRESH, self.HIERARCHICAL_THRESH, self.NMS_THRESH)


class YoloPreProcessor:

    def __init__(self, config):
        self.config = config
        self.vanishing_point_detector = VanishingPointDetector()

    def pre_process(self, image_proto):
        osc_details = apollo_image.OscDetails(image_proto.metadata.id, self.config.osc_api_url)
        image = apollo_image.get_bgr(image_proto.metadata.image_path, osc_details)
        proto_api.add_image_size(image_proto, image.shape)
        vp_processed_image = image
        if self.config.crop_vp:
            detected_vp, confidence = self.vanishing_point_detector.get_vanishing_point(image)
            if detected_vp is not None:
                proto_api.set_vanishing_point(image_proto, detected_vp, confidence)
                vp_processed_image = vp_utils.get_image_fc_vp(detected_vp.y, confidence,
                                                              self.vanishing_point_detector, image)
        pre_processed_image, width_crop_size, height_crop_size = apollo_image.resize_image_fill(vp_processed_image,
                                                                                                self.config.image_height,
                                                                                                self.config.image_width,
                                                                                                3)
        pre_processed_image = cv2.cvtColor(pre_processed_image,
                                           cv2.COLOR_BGR2RGB)
        image_data = ImageData(pre_processed_image,
                               vp_processed_image.shape[1],
                               vp_processed_image.shape[0],
                               width_crop_size,
                               height_crop_size,
                               self.config.image_width,
                               self.config.image_height,
                               None)
        return image_data


class YoloProcessor:

    def __init__(self, config):
        self.yolo_network = YoloNetwork(config.weights_file, config.yolo_network_config,
                                        config.yolo_metadata, config.yolo_classes_names)

    def process(self, image_data_list):
        for i, elem in enumerate(image_data_list):
            result = self.yolo_network.detect(elem.image)
            elem.detections = result
        return image_data_list


class YoloPostProcessor:

    def __init__(self, config):
        self.predict_width = config.image_width
        self.predict_height = config.image_height
        self.algorithm = config.algorithm
        self.algorithm_version = config.algorithm_version
        self.score_thresholds = io_utils.json_load(config.score_thresholds_file)
        self.min_side_size = config.predict_min_side_size

    def validate_detection(self, sign_type, confidence, rect):
        valid_detection = True
        if confidence < self.score_thresholds[sign_type] or \
            rect.br.col - rect.tl.col < self.min_side_size or \
            rect.br.row - rect.tl.row < self.min_side_size:
                valid_detection = False
        return valid_detection

    def post_process(self, image_data, image_proto):
        for detection in image_data.detections:
            rect = self.yolo_bbox_to_roi_rect(detection[2])
            type_name = detection[0]
            confidence = detection[1]
            original_image_rect = self.scale_rect_to_original_image(rect, image_data.image_width,
                                                                    image_data.image_height,
                                                                    image_data.width_crop_size,
                                                                    image_data.height_crop_size,
                                                                    self.predict_width, self.predict_height)
            if self.validate_detection(type_name, confidence, original_image_rect):
                new_roi = image_proto.rois.add()
                new_roi.algorithm = self.algorithm
                new_roi.algorithm_version = self.algorithm_version
                new_roi.manual = False
                new_roi.rect.CopyFrom(original_image_rect)
                new_roi.type = orbb_definitions_pb2.Mark.Value(type_name)
                detection = new_roi.detections.add()
                detection.type = new_roi.type
                detection.confidence = confidence
        return image_proto

    @staticmethod
    def yolo_bbox_to_roi_rect(bounds):
        y_extent = int(bounds[3])
        x_extent = int(bounds[2])
        # Coordinates are around the center
        x_coord = int(bounds[0] - bounds[2] / 2)
        y_coord = int(bounds[1] - bounds[3] / 2)
        rect = orbb_definitions_pb2.Rect()
        rect.tl.col = max(0, x_coord)
        rect.tl.row = max(0, y_coord)
        rect.br.col = x_coord + x_extent
        rect.br.row = y_coord + y_extent
        return rect

    @staticmethod
    def scale_rect_to_original_image(rect, original_width, original_height, width_crop_size, height_crop_size,
                                     scaled_width, scaled_height):
        scale_rows = original_height / (scaled_height - 2 * height_crop_size)
        scale_cols = original_width / (scaled_width - 2 * width_crop_size)
        new_rect = deepcopy(rect)
        new_rect.tl.col = max(0, int((rect.tl.col - width_crop_size) * scale_cols))
        new_rect.tl.row = max(0, int((rect.tl.row - height_crop_size) * scale_rows))
        new_rect.br.row = min(original_width - 1, int((rect.br.row - height_crop_size) * scale_rows))
        new_rect.br.col = min(original_width - 1, int((rect.br.col - width_crop_size) * scale_cols))
        return new_rect