import argparse
import numpy as np
import os
import caffe
import sys
import logging
import logging.config
import cv2
import itertools
import math
import time

import apollo_python_common.image
import apollo_python_common as apollo_python_common
import apollo_python_common.protobuf.orbb_definitions_pb2 as orbb_definitions_pb2
import apollo_python_common.proto_api as proto_api

import traffic_signs_segmentation.utils.configuration as configuration
import traffic_signs_segmentation.utils.network_setup as network_setup
import traffic_signs_segmentation.utils.utils as utils
import traffic_signs_segmentation.utils.image as image_operation

import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util

import apollo_python_common.log_util as log_util
import apollo_python_common.rectangle as rectangle
from vanishing_point.vanishing_point import VanishingPointDetector

conf = configuration.load_configuration()

SEGMENTATION_PROTO_FILE = "../config/deploy.prototxt"
MEAN = "../config/mean.npy"
SEGMENTATION_MODEL = "../config/segmmodel.caffemodel"
MEAN_BLOB = "../config/mean.blob"

# Available on ftp
required_files = [
    SEGMENTATION_MODEL
    ]

VP_DETECTOR = VanishingPointDetector()
VP_CONFIDENCE_THRESHOLD = 0.2
VP_SIGNIFICATIVE_Y_PERCENTAGE = 1.15

class CaffeImageData:

    def __init__(self, image, transformed_image, cropped_image_width, cropped_image_height, width_crop_size, height_crop_size, segmentation_output):
        self.image = image
        self.transformed_image = transformed_image
        self.cropped_image_width = cropped_image_width
        self.cropped_image_height = cropped_image_height
        self.width_crop_size = width_crop_size
        self.height_crop_size = height_crop_size
        self.segmentation_output = segmentation_output

class CaffeModel:

    def __init__(self):
        logger = logging.getLogger(__name__)
        if not utils.exists_paths(required_files):
            logger.error("Paths missing {}".format(required_files))
            sys.exit(-1)
        self.net_segm, self.transformer_segm = network_setup.make_network(
            SEGMENTATION_PROTO_FILE,
            SEGMENTATION_MODEL,
            None)

    def rois_intersect(self, detection_a, detection_b):
        rect_a = rectangle.Rectangle(detection_a.rect[0], detection_a.rect[1],
                                     detection_a.rect[0] + detection_a.rect[2],
                                     detection_a.rect[1] + detection_a.rect[3])
        rect_b = rectangle.Rectangle(detection_b.rect[0], detection_b.rect[1],
                                     detection_b.rect[0] + detection_b.rect[2],
                                     detection_b.rect[1] + detection_b.rect[3])
        intersection_area = rect_a.get_overlapped_rect(rect_b).area()
        if intersection_area:
            percentage = intersection_area / rect_a.area()
            return percentage

    def select_detection(self, detection):
        threshold = conf.class_id2threshold[detection.class_id]
        if threshold < detection.confidence and detection.rect[2] > conf.min_size and detection.rect[3] > conf.min_size:
            return True
        return False

    def non_max_suppression(self, detections):
        if len(detections) == 0:
            return []
        pick = set([i for i in range(len(detections)) if self.select_detection(detections[i])])
        should_continue = True
        while should_continue:
            should_continue = False
            for (i1, i2) in itertools.combinations(list(range(0, len(detections))), 2):
                if i1 in pick and i2 in pick:
                    rect_intersect_percentage1 = self.rois_intersect(detections[i1], detections[i2])
                    rect_intersect_percentage2 = self.rois_intersect(detections[i2], detections[i1])
                    if rect_intersect_percentage1 and (
                            rect_intersect_percentage1 > 0.5 or rect_intersect_percentage2 > 0.5):
                        exclude_idx = i1 if detections[i1].confidence < detections[i2].confidence else i2
                        pick.remove(exclude_idx)
                        should_continue = True
        selected_indices = list(pick)
        selected_detections = [detections[index] for index in selected_indices]
        return selected_detections

    def add_detections(self, image, rects):
        """
        Appends roi detections to metadata
        :param image: roi metadata
        :param rects: np array with signs bounding boxes
        :param class_ids: signs class ids and confidences
        :return:
        """
        for detection_rect in rects:
            roi = image.rois.add()
            roi.algorithm = conf.algorithm
            roi.algorithm_version = conf.algorithm_version
            roi.manual = False
            roi.rect.tl.row = detection_rect.rect[1]
            roi.rect.tl.col = detection_rect.rect[0]
            roi.rect.br.row = detection_rect.rect[1] + detection_rect.rect[3]
            roi.rect.br.col = detection_rect.rect[0] + detection_rect.rect[2]
            if detection_rect.class_id < conf.invalid_id:
                roi.type = conf.class_id2type[detection_rect.class_id]
                detection = roi.detections.add()
                detection.type = roi.type
                detection.confidence = detection_rect.confidence
            else:
                print("error image detection " + image.metadata.image_path)

    def transform_detections(self, rois, image_data):
        """
        Rescales the detections to the original image size
        :param rois: metadata rois
        :param transformer: scale information between original image and network shape
        :return:
        """

        original_width = image_data.cropped_image_width
        original_height = image_data.cropped_image_height
        width_crop_size = image_data.width_crop_size
        height_crop_size = image_data.height_crop_size
        scaled_width = conf.image_size[0]
        scale_height = conf.image_size[1]
        scale_rows = original_height / (scale_height - 2 * height_crop_size)
        scale_cols = original_width / (scaled_width - 2 * width_crop_size)

        for roi in rois:
            roi.rect.tl.col = int(max(0, (roi.rect.tl.col - width_crop_size) * scale_cols))
            roi.rect.tl.row = int(max(0, (roi.rect.tl.row - height_crop_size) * scale_rows))

            roi.rect.br.row = int(
                min((roi.rect.br.row - height_crop_size) * scale_rows, original_height - 1))
            roi.rect.br.col = int(min((roi.rect.br.col - width_crop_size) * scale_cols
                                      , original_width - 1))

    def convert_detections_to_rois(self, image_data, image_proto):
            mask = np.argmax(image_data.segmentation_output, axis=0)
            rects = image_operation.get_rects(mask)
            for rect in rects:
                mask_for_class_id = image_data.segmentation_output[rect.class_id][rect.rect[1]:rect.rect[1] + rect.rect[3],
                                    rect.rect[0]:rect.rect[0] + rect.rect[2]]
                rect.confidence = np.average(mask_for_class_id)
            rects = self.non_max_suppression(rects)
            self.add_detections(image_proto, rects)
            self.transform_detections(image_proto.rois, image_data)
            return image_proto

    def get_image_fc_VP(self, image):
        '''
        Gets the area of interest where traffic signs are located in a image
        :param image: full image
        :return: cropped image with the area above vanishing point
        '''
        detected_vp, confidence = VP_DETECTOR.get_vanishing_point(image)
        if confidence > VP_CONFIDENCE_THRESHOLD:
            crop_y = math.floor(detected_vp.y * VP_SIGNIFICATIVE_Y_PERCENTAGE)
            new_image = image[:crop_y, :]
            return new_image
        else:
            return image

    def segmentation_pre_process(self, image_proto):
        image = apollo_python_common.image.get_bgr(image_proto.metadata.image_path)
        proto_api.add_image_size(image_proto, image.shape)
        vp_processed_image = self.get_image_fc_VP(image)
        cropped_image, width_crop_size, height_crop_size = apollo_python_common.image.resize_image_fill(vp_processed_image,
                                                                                                        conf.image_size[1],
                                                                                                        conf.image_size[0], 3)
        transformed_image = self.transformer_segm.preprocess("data", cropped_image)
        image_preprocessed = CaffeImageData(image, transformed_image, vp_processed_image.shape[1], vp_processed_image.shape[0], width_crop_size, height_crop_size, None)
        return image_preprocessed

    def segmentation_predict(self, image_data_list):
        caffe.set_mode_gpu()
        for i, elem in enumerate(image_data_list):
            self.net_segm.blobs[self.net_segm.inputs[0]].data[i] = elem.transformed_image
        segmentation_output = self.net_segm.forward()[self.net_segm.outputs[0]]
        for i, elem in enumerate(image_data_list):
            elem.segmentation_output = segmentation_output[i]
        return image_data_list

    def post_process(self, image_data, image_proto):
        self.convert_detections_to_rois(image_data, image_proto)
        return image_proto




