import argparse
import numpy as np
import os
import caffe
import sys
import itertools
import threading
import time


import orbb_metadata_pb2, orbb_definitions_pb2
import apollo_python_common.proto_api as proto_api
import batch
import utils
import network_setup
import image as image_operation
import configuration
import logging
import logging.config
import apollo_python_common.log_util as log_util
import apollo_python_common.rectangle as rectangle


# TODO: add logging
conf = configuration.load_configuration()

SEGMENTATION_PROTO_FILE = "./config/deploy.prototxt"
SEGMENTATION_MODEL = "/config/segmmodel.caffemodel"
# Available on ftp
required_files = [
    SEGMENTATION_PROTO_FILE,
    SEGMENTATION_MODEL
    ]

class BatchHandler():
    """
    Handles a batch of images and produces a protobuf metadata with the detections
    """

    def __init__(self, detector, lock):
        """
        :param detector: lambda function that detects traffic signs
        :param classifier: lambda function that classifies a traffic sign
        """
        self.metadata = orbb_metadata_pb2.ImageSet()
        self.metadata.name = "caffe"
        self.detector = detector
        self.lock = lock

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
                    if rect_intersect_percentage1 and (rect_intersect_percentage1 > 0.7 or rect_intersect_percentage2 > 0.7):
                        exclude_idx = i1 if detections[i1].confidence < detections[i2].confidence else i2
                        pick.remove(exclude_idx)
                        should_continue = True
        selected_indices = list(pick)
        selected_detections = [detections[index] for index in selected_indices]
        return selected_detections

    def convert_detections_to_rois(self, image_batch, net_output):
        """
        Converts the detections to protobuf rois
        :param image_batch: batch of images
        :param net_output: segmentation result
        :return:
        """
        for i, elem in enumerate(image_batch):
            # skip processing invalid images
            if elem.image_data is None:
                continue
            mask = np.argmax(net_output[i], axis=0)
            rects = image_operation.get_rects(mask, os.path.basename(elem.image_path))

            for rect in rects:
                mask_for_class_id = net_output[i][rect.class_id][rect.rect[1]:rect.rect[1]+rect.rect[3], rect.rect[0]:rect.rect[0]+rect.rect[2]]
                rect.confidence = np.average(mask_for_class_id)

            image = self.metadata.images.add()
            image.metadata.image_path = os.path.basename(elem.image_path)
            image.metadata.region = ""
            image.metadata.trip_id = ""
            image.metadata.image_index = 0

            rects = self.non_max_suppression(rects)
            add_detections(image, rects)
            shape = elem.image_data.shape
            transform_detections(image.rois, elem.scale_factors, shape)


    def __call__(self, image_batch):
        """
        Handles a batch of images and generates detections
        :param image_batch:
        :return:
        """
        self.lock.acquire()
        out = self.detector(image_batch)
        self.lock.release()
        self.convert_detections_to_rois(image_batch, out)
        
    def get_result(self):
        """
        Returns protobuf metadata
        :return: protobuf metadata
        """
        return self.metadata


def detect_batch(net, current_batch):
    """
    Segmentation network forward pass
    :param net: Segmentation network
    :param current_batch: image batch
    :return: segmentation output
    """
    caffe.set_mode_gpu()

    for i, elem in enumerate(current_batch):
        # skip processing invalid images
        if elem.image_data is not None:
            net.blobs[net.inputs[0]].data[i] = elem.image_data
    result = net.forward()[net.outputs[0]]
    return result


def add_detections(image, rects):
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


def transform_detections(rois, transformer, shape):
    """
    Rescales the detections to the original image size
    :param rois: metadata rois
    :param transformer: scale information between original image and network shape
    :param shape: network input image shape
    :return:
    """

    original_width = transformer.width
    original_height = transformer.height
    width_crop_size = transformer.width_crop_size
    height_crop_size = transformer.height_crop_size

    scaled_width = conf.image_size[0]
    scale_height = conf.image_size[1]

    scale_rows = original_height / (scale_height - 2 * height_crop_size)
    scale_cols = original_width / (scaled_width - 2 * width_crop_size)

    for roi in rois:
        roi.rect.tl.col = int(max(0, (roi.rect.tl.col - width_crop_size )* scale_cols))
        roi.rect.tl.row = int(max(0, (roi.rect.tl.row - height_crop_size) * scale_rows))

        roi.rect.br.row = int(min((roi.rect.br.row - height_crop_size ) * scale_rows, original_height - 1))
        roi.rect.br.col = int(min((roi.rect.br.col - width_crop_size) * scale_cols
                                  , original_width - 1))




class InferenceService:
    """
    Inference Service class
    """

    def __init__(self):
        self.net_segm, self.transformer_segm = network_setup.make_network(
            SEGMENTATION_PROTO_FILE,
            SEGMENTATION_MODEL,
            None)

    def process(self, images_path, lock):
        """
        Server process method runs segmentation and classification and returns the rois metadata
        :param images_path: images path folder
        :return: rois metadata
        """
        logger = logging.getLogger(__name__)
        if not utils.exists_paths([images_path]):
            logger.info("Request path missing {}.".format(images_path))
            return None

        detector = lambda x: detect_batch(self.net_segm, x)

        batch_reader = batch.BatchReader(
            images_path,
            self.transformer_segm,
            network_setup.get_batch_size(self.net_segm), lock)
        batch_handler = BatchHandler(detector, lock)
        batch_processor = batch.BatchProcessor(batch_handler)
        batch_reader.start()
        batch_processor.start()
        batch_reader.join()
        batch_processor.join()
        return batch_handler.get_result()


if __name__ == "__main__":
    log_util.config(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_path", type=str, required=True)
    parser.add_argument(
        "-o", "--output_path", type=str, required=False, default="./")

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    if not utils.exists_paths(required_files):
        logger.error("Paths missing {}".format(required_files))
        sys.exit(-1)
    lock = threading.Lock()
    server = InferenceService()
    response = server.process(args.input_path, lock)
    proto_api.serialize_proto_instance(response, args.output_path)



