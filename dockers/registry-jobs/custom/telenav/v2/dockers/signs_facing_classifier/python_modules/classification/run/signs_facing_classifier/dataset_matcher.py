import argparse
import logging
import os

import cv2
import pandas as pd

from apollo_python_common import log_util, proto_api, io_utils
from apollo_python_common.rectangle import Rectangle
from classification.scripts.signs_facing_classifier.constants import RoiDfColumn
from classification.scripts.signs_facing_classifier.constants import SignFacingLabel
import classification.scripts.signs_facing_classifier.data_common as data_common


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--match_cfg_json", help="path to json containing the matcher params",
                        type=str, required=True)

    return parser.parse_args()


class SignFacingDatasetMatcher:
    """ The SingsFacingDatasetMatcher class can be used to match the ground truth ROIs with the predictions from
     any given object detector, as long as the predictions are in protobuf format. The signs facing dataset, containing
     the ground truth is the raw image data tagged in front, left and right directories. """

    FULL_SET_DIR = "full_set_dir"
    NEW_ORIG_IMG_DIR = "new_orig_img_dir"
    OLD_ORIG_IMG_DIR = "old_orig_img_dir"
    NEW_DETECTIONS_FILE = "new_detections_file"
    OLD_DETECTIONS_FILE = "old_detections_file"
    TRAIN_SET_DIR = "train_set_dir"
    TEST_SET_DIR = "test_set_dir"
    IOU_THRESHOLD = "iou_threshold"
    SPLIT_PERCENTAGE = "split_percentage"
    MATCHED_RESULTS_FILE = "matched_results_file"

    def __init__(self, config):
        self.full_set_dir = config[self.FULL_SET_DIR]
        self.new_orig_img_dir = config[self.NEW_ORIG_IMG_DIR]
        self.old_orig_img_dir = config[self.OLD_ORIG_IMG_DIR]
        self.new_detections_file = config[self.NEW_DETECTIONS_FILE]
        self.old_detections_file = config[self.OLD_DETECTIONS_FILE]
        self.train_set_dir = config[self.TRAIN_SET_DIR]
        self.test_set_dir = config[self.TEST_SET_DIR]
        self.matched_results_file = config[self.MATCHED_RESULTS_FILE]
        self.split_percentage = config[self.SPLIT_PERCENTAGE]
        self.iou_threshold = config[self.IOU_THRESHOLD]
        self.gt_df = self._load_gt_data()
        self.pred_df = self._load_all_preds()

    def _get_orig_img_path(self, img_name):
        """ Given an original image name without extension, it returns the full path (including directories) of
         the image. """

        jpg = '{}.jpg'.format(img_name)
        jpeg = '{}.jpeg'.format(img_name)

        new_jpg_path = os.path.join(self.new_orig_img_dir, jpg)
        new_jpeg_path = os.path.join(self.new_orig_img_dir, jpeg)
        old_jpg_path = os.path.join(self.old_orig_img_dir, jpg)
        old_jpeg_path = os.path.join(self.old_orig_img_dir, jpeg)

        if os.path.isfile(new_jpg_path):
            return new_jpg_path
        elif os.path.isfile(new_jpeg_path):
            return new_jpeg_path
        elif os.path.isfile(old_jpg_path):
            return old_jpg_path
        else:
            return old_jpeg_path

    @staticmethod
    def _get_img_name(image_proto):
        """ Returns the image name from a given protobuf definition. """
        return os.path.basename(image_proto.metadata.image_path).split(".")[0]

    @staticmethod
    def _load_preds_to_df(pred_metadata):
        """ Loads the object detection model predictions into a dataframe for processing. """

        images = proto_api.read_imageset_file(pred_metadata).images
        preds_dict = {RoiDfColumn.IMG_NAME_COL: [], RoiDfColumn.ROI_CLASS_COL: [], RoiDfColumn.TL_ROW_COL: [],
                      RoiDfColumn.TL_COL_COL: [], RoiDfColumn.BR_ROW_COL: [], RoiDfColumn.BR_COL_COL: []}

        for img_proto in images:
            img_name = SignFacingDatasetMatcher._get_img_name(img_proto)
            for roi in img_proto.rois:
                preds_dict[RoiDfColumn.IMG_NAME_COL].append(img_name)
                preds_dict[RoiDfColumn.ROI_CLASS_COL].append(proto_api.get_roi_type_name(roi.type))
                preds_dict[RoiDfColumn.TL_ROW_COL].append(roi.rect.tl.row)
                preds_dict[RoiDfColumn.TL_COL_COL].append(roi.rect.tl.col)
                preds_dict[RoiDfColumn.BR_ROW_COL].append(roi.rect.br.row)
                preds_dict[RoiDfColumn.BR_COL_COL].append(roi.rect.br.col)

        return pd.DataFrame(preds_dict)

    def _load_all_preds(self):
        """ Loading all the predictions obtained by the object detector. """

        logger.info("loading detections for the old dataset {} ...".format(self.old_detections_file))
        old_preds = self._load_preds_to_df(self.old_detections_file)
        logger.info("old predictions count: {}".format(len(old_preds)))

        logger.info("loading detections for the new dataset {} ...")
        new_preds = self._load_preds_to_df(self.new_detections_file)
        logger.info("new predictions count: {}".format(len(new_preds)))

        all_pred_df = pd.concat([old_preds, new_preds]).drop_duplicates(keep='last')\
            .sort_values([RoiDfColumn.IMG_NAME_COL, RoiDfColumn.ROI_CLASS_COL])
        logger.info("full predictions count: {}".format(len(all_pred_df)))
        logger.info(all_pred_df.head())

        return all_pred_df

    def _load_gt_data(self):
        """ Loading all the ground truth data as a dataframe. """

        logger.info("loading the ground truth data...")
        gt_data_df = pd.concat(list(data_common.load_imgs_in_dataframes(self.full_set_dir, with_area=False)))\
            .sort_values([RoiDfColumn.IMG_NAME_COL, RoiDfColumn.ROI_CLASS_COL])
        logger.info("ground truth count: {}".format(len(gt_data_df)))
        logger.info(gt_data_df.head())

        return gt_data_df

    @staticmethod
    def _get_preds_by_class_and_img(class_name, img_name, pred_df):
        """ Get matching predictions by roi class name and image name. """
        return pred_df[(pred_df[RoiDfColumn.IMG_NAME_COL] == img_name) & (pred_df[RoiDfColumn.ROI_CLASS_COL] == class_name)]

    def _match_predictions(self, gt_row):
        """ Match the predictions from the object detector with a given ground truth row. """
        gt_rect = Rectangle(gt_row[RoiDfColumn.TL_COL_COL], gt_row[RoiDfColumn.TL_ROW_COL],
                            gt_row[RoiDfColumn.BR_COL_COL], gt_row[RoiDfColumn.BR_ROW_COL])
        prematch_df = self._get_preds_by_class_and_img(gt_row[RoiDfColumn.ROI_CLASS_COL],
                                                       gt_row[RoiDfColumn.IMG_NAME_COL], self.pred_df)

        ious = {}
        for idx, row in prematch_df.iterrows():
            pred_rect = Rectangle(row[RoiDfColumn.TL_COL_COL], row[RoiDfColumn.TL_ROW_COL], row[RoiDfColumn.BR_COL_COL],
                                  row[RoiDfColumn.BR_ROW_COL])
            iou = gt_rect.intersection_over_union(pred_rect)
            ious[iou] = (row[RoiDfColumn.TL_COL_COL], row[RoiDfColumn.TL_ROW_COL], row[RoiDfColumn.BR_COL_COL],
                         row[RoiDfColumn.BR_ROW_COL])

        if len(ious) == 0:
            return None

        matched_iou = max(ious.keys())
        if matched_iou > self.iou_threshold:
            return ious[matched_iou]
        else:
            return None

    def _update_gt(self):
        """ Update the ROI coordinates for the ground truth items that were matched, add an IS_MATCHED column,
         to be able to tell how many of the gt items were matched with predictions. """
        self.gt_df[RoiDfColumn.TL_COL_COL] = self.gt_df.apply(
            lambda row: row[RoiDfColumn.MATCHED_COL][0] if row[RoiDfColumn.MATCHED_COL] is not None else row[
                RoiDfColumn.TL_COL_COL], axis=1)
        self.gt_df[RoiDfColumn.TL_ROW_COL] = self.gt_df.apply(
            lambda row: row[RoiDfColumn.MATCHED_COL][1] if row[RoiDfColumn.MATCHED_COL] is not None else row[
                RoiDfColumn.TL_ROW_COL], axis=1)
        self.gt_df[RoiDfColumn.BR_COL_COL] = self.gt_df.apply(
            lambda row: row[RoiDfColumn.MATCHED_COL][2] if row[RoiDfColumn.MATCHED_COL] is not None else row[
                RoiDfColumn.BR_COL_COL], axis=1)
        self.gt_df[RoiDfColumn.BR_ROW_COL] = self.gt_df.apply(
            lambda row: row[RoiDfColumn.MATCHED_COL][3] if row[RoiDfColumn.MATCHED_COL] is not None else row[
                RoiDfColumn.BR_ROW_COL], axis=1)

        self.gt_df[RoiDfColumn.IS_MATCHED_COL] = self.gt_df[RoiDfColumn.MATCHED_COL].apply(
            lambda coords: True if coords is not None else False)

        self.gt_df = self.gt_df.drop([RoiDfColumn.MATCHED_COL], axis=1)

    @staticmethod
    def _get_crop_path(row, dest_path):
        """ Based on a given roi dataframe row and a destination path, returns the corresponding crop file path. """
        coords = row[RoiDfColumn.TL_COL_COL], row[RoiDfColumn.TL_ROW_COL], row[RoiDfColumn.BR_COL_COL], row[
            RoiDfColumn.BR_ROW_COL]

        crop_name = row[RoiDfColumn.IMG_NAME_COL]
        for coord in coords:
            crop_name = f"{crop_name}_{coord}"

        crop_path = os.path.join(dest_path, row[RoiDfColumn.ORIENTATION_COL])

        return os.path.join(crop_path, '{}_{}.jpg'.format(crop_name, row[RoiDfColumn.ROI_CLASS_COL]))

    def _save_crop(self, row):
        """ Given a row from a roi dataframe saves the corresponding crop file to disk. """
        img_path = self._get_orig_img_path(row[RoiDfColumn.IMG_NAME_COL])
        img = cv2.imread(img_path)

        crop_img = img[row[RoiDfColumn.TL_ROW_COL]: row[RoiDfColumn.BR_ROW_COL],
                   row[RoiDfColumn.TL_COL_COL]: row[RoiDfColumn.BR_COL_COL]]

        cv2.imwrite(row[RoiDfColumn.CROP_PATH_COL], crop_img)

    def _save_crops(self, data_df, dest_path):
        """ Given a ROI dataframe saves the corresponding cropped ROIs to the given destination path. """
        logger.info("saving the cropped ROIs to {}... ".format(dest_path))
        to_save_df = data_df.sort_values([RoiDfColumn.ORIENTATION_COL])
        to_save_df[RoiDfColumn.CROP_PATH_COL] = to_save_df.apply(lambda row: self._get_crop_path(row, dest_path), axis=1)
        to_save_df.apply(lambda row: self._save_crop(row), axis=1)

    def run_matching(self):
        """ Runs the prediction to ground truth matching. """
        logger.info("run the prediction to ground truth matching...")

        self.gt_df[RoiDfColumn.MATCHED_COL] = self.gt_df.apply(lambda gt_row: self._match_predictions(gt_row), axis=1)
        self._update_gt()

        logger.info("matching - gt size: {}".format(len(self.gt_df)))
        logger.info(self.gt_df.head())
        logger.info("count of matched items...")
        logger.info(self.gt_df.groupby([RoiDfColumn.IS_MATCHED_COL]).count())
        self.gt_df.to_csv(self.matched_results_file, index=False)

    def split_and_save(self):
        """ Splits the matched ground truth data frame into train and testing data sets and saves the corresponding
         crops to disk. """
        logger.info("matched ground truth size: ".format(len(self.gt_df)))
        train_df, test_df = data_common.train_test_split_stratified(self.gt_df, self.split_percentage,
                                                                    RoiDfColumn.ORIENTATION_COL)
        logger.info("train set size: ".format(len(train_df)))
        logger.info("test set size: ".format(len(test_df)))

        io_utils.create_folder(os.path.join(self.test_set_dir, SignFacingLabel.CLS_FRONT))
        io_utils.create_folder(os.path.join(self.test_set_dir, SignFacingLabel.CLS_LEFT))
        io_utils.create_folder(os.path.join(self.test_set_dir, SignFacingLabel.CLS_RIGHT))

        io_utils.create_folder(os.path.join(self.train_set_dir, SignFacingLabel.CLS_FRONT))
        io_utils.create_folder(os.path.join(self.train_set_dir, SignFacingLabel.CLS_LEFT))
        io_utils.create_folder(os.path.join(self.train_set_dir, SignFacingLabel.CLS_RIGHT))

        self._save_crops(train_df, self.train_set_dir)
        self._save_crops(test_df, self.test_set_dir)


if __name__ == '__main__':
    log_util.config(__file__)
    logger = logging.getLogger(__name__)

    args = parse_arguments()

    dataset_matcher = SignFacingDatasetMatcher(io_utils.json_load(args.match_cfg_json))

    dataset_matcher.run_matching()
    dataset_matcher.split_and_save()
