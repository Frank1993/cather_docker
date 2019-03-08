import os
import cv2
import argparse
import logging
from copy import deepcopy
from multiprocessing import Pool
from collections import defaultdict

import orbb_metadata_pb2
import apollo_python_common.log_util as log_util
import apollo_python_common.io_utils as io_utils
import apollo_python_common.proto_api as roi_metadata
import apollo_python_common.image as image

import tools.vp_utils as vp_utils

class YoloData:
    def __init__(self, transformed_image, cropped_image_width, cropped_image_height, width_crop_size,
                 height_crop_size, vp_y):
        self.transformed_image = transformed_image
        self.cropped_image_width = cropped_image_width
        self.cropped_image_height = cropped_image_height
        self.width_crop_size = width_crop_size
        self.height_crop_size = height_crop_size
        self.vp_y = vp_y


class YoloDetectionGenerator:

    @staticmethod
    def generate_type2class_id(labels_path):
        type_class_dict = io_utils.json_load(labels_path)
        class_id = 1
        type2class_id = {}
        for key, value in type_class_dict.items():
            type2class_id[key] = class_id
            class_id += 1
        return type2class_id

    def __init__(self, detection_path, images_file, image_width, image_height, min_side_size, crop_vp, labels_path):
        self.detection_path = detection_path
        self.images_file = images_file
        self.image_width = image_width
        self.image_height = image_height
        self.min_side_size = min_side_size
        self.crop_vp = crop_vp
        self.labels_path = labels_path
        self.type2class_id = self.generate_type2class_id(labels_path)

    def transform_roi_from_default_to_resize(self, roi, image_data):
        original_width = image_data.cropped_image_width
        original_height = image_data.cropped_image_height
        width_crop_size = image_data.width_crop_size
        height_crop_size = image_data.height_crop_size
        scaled_width = self.image_width
        scale_height = self.image_height
        scale_rows = original_height / (scale_height - 2 * height_crop_size)
        scale_cols = original_width / (scaled_width - 2 * width_crop_size)
        new_roi = deepcopy(roi)
        new_roi.tl_col = int((roi.tl_col / scale_cols + width_crop_size))
        new_roi.tl_row = int((roi.tl_row / scale_rows + height_crop_size))
        new_roi.br_row = int((roi.br_row / scale_rows + height_crop_size))
        new_roi.br_col = int((roi.br_col / scale_cols + width_crop_size))
        return new_roi

    def __call__(self, image_rois):
            img = image.get_bgr(image_rois[0])
            if img is not None:
                img_path = os.path.join(self.detection_path,
                                        os.path.basename(image_rois[0]))
                if self.crop_vp:
                    vp_processed_image = vp_utils.get_image_fc_vp(img)
                else:
                    vp_processed_image = img
                cropped_image, width_crop_size, height_crop_size = image.resize_image_fill(vp_processed_image,
                                                                                           self.image_height,
                                                                                           self.image_width,
                                                                                           3)
                vp_y = img.shape[0] - vp_processed_image.shape[0]
                image_preprocessed = YoloData(cropped_image, vp_processed_image.shape[1],
                                              vp_processed_image.shape[0], width_crop_size, height_crop_size, vp_y)
                cv2.imwrite(img_path, image_preprocessed.transformed_image)
                detection_file = self.detection_path
                detection_file += io_utils.get_filename(img_path) + ".txt"

                with open(self.images_file, "a+") as images_file:
                    images_file.write(img_path)
                    images_file.write("\n")

                with open(detection_file, "w+") as label_file:
                    for roi in image_rois[1]:
                        new_roi = self.transform_roi_from_default_to_resize(roi, image_preprocessed)
                        new_roi_width = new_roi.br_col - new_roi.tl_col
                        new_roi_height = new_roi.br_row - new_roi.tl_row
                        if new_roi_height > self.min_side_size and new_roi_width > self.min_side_size:
                            x = ((new_roi.tl_col + new_roi.br_col) // 2) / self.image_width
                            y = ((new_roi.tl_row + new_roi.br_row) // 2) / self.image_height
                            width = (new_roi.br_col - new_roi.tl_col) / self.image_width
                            height = (new_roi.br_row - new_roi.tl_row) / self.image_height
                            class_id = self.type2class_id[str(roi.r_type)]
                            label_file.write("{} {} {} {} {}".format(class_id, x, y, width, height))
                            label_file.write("\n")


class Roi:
    """
    Stub class for metadata Roi class
    """
    def __init__(self, orbb_roi):
        self.r_type = orbb_roi.type
        self.tl_row = orbb_roi.rect.tl.row
        self.br_row = orbb_roi.rect.br.row
        self.tl_col = orbb_roi.rect.tl.col
        self.br_col = orbb_roi.rect.br.col
        self.false_pos = (orbb_roi.validation
                          == orbb_metadata_pb2.FALSE_POSITIVE)


def load_all_rois(input_path):
    metadata = roi_metadata.read_imageset_file(input_path)
    metadata = roi_metadata.remove_duplicate_rois(metadata)
    rois = defaultdict(list)
    for image_proto in metadata.images:
        current_image = os.path.normpath(
            os.path.join(os.path.dirname(input_path), os.path.basename(image_proto.metadata.image_path)))
        if not os.path.exists(current_image):
            continue
        rois[current_image] = list()
        for roi in image_proto.rois:
            rois[current_image].append(Roi(roi))
    return rois


def do_work(generator, payload):
    pool = Pool(20)
    pool.map(generator, payload.items())


def main():

    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    logger.info('Prepare yolo dataset')

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",
                        type=str, required=True)
    parser.add_argument("--output_path",
                        type=str, required=True)
    parser.add_argument("--images_file",
                        type=str, required=True)
    parser.add_argument("--images_width",
                        type=int, required=True)
    parser.add_argument("--images_height",
                        type=int, required=True)
    parser.add_argument("--min_side",
                        type=int, required=True)
    parser.add_argument("--classes_ids",
                        type=str, required=True)
    parser.add_argument("--crop_vp",
                        type=bool, required=True)

    args = parser.parse_args()

    rois = load_all_rois(args.input_path)
    io_utils.create_folder(args.output_path)
    do_work(YoloDetectionGenerator(args.output_path, args.images_file, args.images_width, args.images_height,
                                   args.min_side, args.crop_vp, args.classes_ids), rois)


if __name__ == "__main__":
    main()