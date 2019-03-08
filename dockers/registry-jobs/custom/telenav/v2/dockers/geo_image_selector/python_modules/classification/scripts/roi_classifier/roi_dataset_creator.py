import os

from tqdm import tqdm

tqdm.pandas()
import numpy as np
import PIL

import multiprocessing
from multiprocessing import Pool
import random

import apollo_python_common.io_utils as io_utils

import apollo_python_common.proto_api as proto_api
import apollo_python_common.image as image


class ROIDatasetCreator:
    GT_FOLDER = "gt_folder"
    PRED_FILE = "predictions_rois_file"
    IOU_THRESHOLD = "iou_threshold"
    MIN_SIZE = "min_size"
    DATASET_CLASSES_FILE = "dataset_classes_file"
    DELTA_PERC_ARR = "delta_perc_arr"
    BASE_OUTPUT_PATH = "base_output_path"
    RAW_IMGS_FOLDER = "raw_imgs"
    GOOD_FOLDER_NAME = "good"
    BAD_FOLDER_NAME = "bad"
    SELECTED_CLASSES_KEY = "selected_classes"

    def __init__(self, config):
        self.config = config
        self.gt_rois_dict = self.__get_images_dict(os.path.join(self.config[self.GT_FOLDER], "rois.bin"))
        self.pred_rois_dict = self.__get_images_dict(self.config[self.PRED_FILE])
        self.dataset_classes = io_utils.json_load(self.config[self.DATASET_CLASSES_FILE])[self.SELECTED_CLASSES_KEY]

    def __get_images_dict(self, path):
        return proto_api.create_images_dictionary(proto_api.read_imageset_file(path))

    def __get_roi_dict(self, rois_list):
        return {(proto_api.get_rect_from_roi(roi)): roi for roi in rois_list}

    def __get_tp_rois(self, gt_rois, pred_rois):
        found_pred_roi = set()
        for (pred_label, pred_rect), pred_roi in pred_rois.items():
            for (gt_label, gt_rect), gt_roi in gt_rois.items():
                if gt_rect.intersection_over_union(pred_rect) > self.config[self.IOU_THRESHOLD]:
                    found_pred_roi.add((pred_label, pred_rect))
                    break
        return found_pred_roi

    def __filter_rois_by_size(self, rois, pred_rois):
        filtered_rois = list()
        for pred_label, pred_rect in rois:
            if pred_rect.width() > self.config[self.MIN_SIZE] and pred_rect.height() > self.config[self.MIN_SIZE]:
                filtered_rois.append(pred_rois[(pred_label, pred_rect)])
        return filtered_rois

    def __filter_rois_by_class(self, all_rois):
        if len(self.dataset_classes) == 0:
            return all_rois

        selected_rois = list()
        for roi in all_rois:
            label, rect = proto_api.get_rect_from_roi(roi)
            if label in self.dataset_classes:
                selected_rois.append(roi)
        return selected_rois

    def __filter_rois(self, rois, pred_rois):
        filtered_rois = self.__filter_rois_by_size(rois, pred_rois)
        filtered_rois = self.__filter_rois_by_class(filtered_rois)

        return filtered_rois

    def __get_preds(self, gt_rois_list, pred_rois_list):
        gt_rois = self.__get_roi_dict(gt_rois_list)
        pred_rois = self.__get_roi_dict(pred_rois_list)

        tp_rois = self.__get_tp_rois(gt_rois, pred_rois)
        fp_rois = set(pred_rois.keys()) - tp_rois

        true_positives = self.__filter_rois(tp_rois, pred_rois)
        false_positives = self.__filter_rois(fp_rois, pred_rois)

        return true_positives, false_positives

    def __get_cropped_rois(self, image, pred_rois):

        cropped_rois = []

        image_height, image_width, _ = image.shape

        for roi in pred_rois:
            label_name, rect = proto_api.get_rect_from_roi(roi)

            tl_col = int(rect.xmin)
            tl_row = int(rect.ymin)
            br_col = int(rect.xmax)
            br_row = int(rect.ymax)

            roi_width = br_col - tl_col
            roi_height = br_row - tl_row

            for delta_perc in self.config[self.DELTA_PERC_ARR]:
                delta_w = int(delta_perc * roi_width) // 2
                delta_h = int(delta_perc * roi_height) // 2

                new_tl_row = max(0, tl_row - delta_h)
                new_br_row = min(image_height, br_row + delta_h)

                new_tl_col = max(0, tl_col - delta_w)
                new_br_col = min(image_width, br_col + delta_w)

                cr = image[new_tl_row:new_br_row, new_tl_col:new_br_col]

                cropped_rois.append((cr, rect, roi.type))

        return cropped_rois

    def __save_cropped_rois(self, rois, file_name, output_folder):

        output_path = os.path.join(self.config[self.BASE_OUTPUT_PATH], self.RAW_IMGS_FOLDER, output_folder)
        io_utils.create_folder(output_path)

        for cr, rect, _ in rois:
            save_file_name = "1_{}_{}_{}-{}-{}-{}.jpg".format(random.randint(1, 9000000),
                                                              os.path.basename(file_name).split(".")[0],
                                                              rect.xmin, rect.ymin, rect.xmax, rect.ymax)
            path = os.path.join(output_path, save_file_name)
            PIL.Image.fromarray(cr).save(path)

    def save_rois_to_disk(self, files):

        for file_name in tqdm(list(files)):
            true_positives, false_positives = self.__get_preds(self.gt_rois_dict.get(file_name, []),
                                                               self.pred_rois_dict.get(file_name, []))

            img_path = os.path.join(self.config[self.GT_FOLDER], file_name)

            if not os.path.isfile(img_path):
                print("Image not found {}".format(img_path))
                continue

            img = image.get_rgb(img_path)

            true_cropped_rois = self.__get_cropped_rois(img, true_positives)
            false_cropped_rois = self.__get_cropped_rois(img, false_positives)

            self.__save_cropped_rois(true_cropped_rois, file_name, self.GOOD_FOLDER_NAME)
            self.__save_cropped_rois(false_cropped_rois, file_name, self.BAD_FOLDER_NAME)

    def create_dataset(self):
        nr_threads = multiprocessing.cpu_count() // 2
        all_files = list(set(self.gt_rois_dict.keys()).union(set(self.pred_rois_dict.keys())))
        all_files_array = np.array_split(all_files, nr_threads)
        pool = Pool(nr_threads)
        pool.map(self.save_rois_to_disk, all_files_array)
