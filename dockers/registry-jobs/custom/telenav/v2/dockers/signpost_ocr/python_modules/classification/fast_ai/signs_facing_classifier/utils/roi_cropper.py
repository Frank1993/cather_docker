import argparse
import logging
import os

import cv2
import pandas as pd

from apollo_python_common import log_util, io_utils, image
from classification.fast_ai.signs_facing_classifier.utils.constants import RoiDfColumn, SignFacingLabel
from classification.fast_ai.signs_facing_classifier.utils import data_common


class ROICropper:
    """ Class used for cropping ROIS with a given factor. """
    ORIG_IMG_COL = 'orig_img'

    def __init__(self, params):
        self.params = params

    def _load_to_df(self):
        """Load the image paths from the input dir into a dataframe. """

        img_df = pd.concat(list(data_common.load_imgs_in_dataframes(self.params.input_dir, with_area=False)))
        logger.info("loading input images to dataframe...")
        logger.info(img_df.head())

        new_imgs, old_imgs = data_common.get_orig_images(self.params.new_orig_img_dir, self.params.old_orig_img_dir)
        img_df[self.ORIG_IMG_COL] = img_df[RoiDfColumn.IMG_NAME_COL].apply(lambda img: data_common.get_orig_img_path(
            img, new_imgs, old_imgs, self.params.new_orig_img_dir, self.params.old_orig_img_dir))

        return img_df

    def _make_output_dirs(self):
        """ Create the crop output directories, if they do not exist. """
        io_utils.create_folder(os.path.join(self.params.output_dir, SignFacingLabel.FRONT))
        io_utils.create_folder(os.path.join(self.params.output_dir, SignFacingLabel.LEFT))
        io_utils.create_folder(os.path.join(self.params.output_dir, SignFacingLabel.RIGHT))

    def _save_cropped_roi(self, roi_row, cropped_roi):
        """ Saves the cropped ROI image as a jpg file to the corresponding output folder. """
        img_name = '{}_{}_{}_{}_{}_{}.jpg'.format(roi_row[RoiDfColumn.IMG_NAME_COL], roi_row[RoiDfColumn.TL_COL_COL],
                                                  roi_row[RoiDfColumn.TL_ROW_COL], roi_row[RoiDfColumn.BR_ROW_COL],
                                                  roi_row[RoiDfColumn.BR_ROW_COL], roi_row[RoiDfColumn.ROI_CLASS_COL])
        output_dir = os.path.join(self.params.output_dir, roi_row[RoiDfColumn.ORIENTATION_COL])
        crop_path = os.path.join(output_dir, img_name)
        logger.info("saving image: {}".format(crop_path))

        cv2.imwrite(crop_path, cv2.cvtColor(cropped_roi, cv2.COLOR_RGB2BGR))

    def _crop_square_roi(self, roi_row):
        """ Given a data frame row it creates a square crop from the original and saves it. """
        full_img = image.get_rgb(roi_row[self.ORIG_IMG_COL])
        crop = image.crop_square_roi(full_img, roi_row[RoiDfColumn.TL_COL_COL], roi_row[RoiDfColumn.TL_ROW_COL],
                                     roi_row[RoiDfColumn.BR_COL_COL], roi_row[RoiDfColumn.BR_ROW_COL],
                                     self.params.sq_crop_factor)

        self._save_cropped_roi(roi_row, crop)

    def make_square_crops(self):
        """ Run the square cropping over all the images in the input directory. """
        img_df = self._load_to_df()
        self._make_output_dirs()

        logger.info("starting to crop {} images...".format(len(img_df)))
        img_df.apply(lambda row: self._crop_square_roi(row), axis=1)

        logger.info("finished cropping images.")


def run(cropper_config):
    train_params = cropper_config
    train_params.input_dir = cropper_config.train_input_dir
    train_params.output_dir = cropper_config.train_output_dir
    ROICropper(train_params).make_square_crops()

    test_params = cropper_config
    test_params.input_dir = cropper_config.test_input_dir
    test_params.output_dir = cropper_config.test_output_dir
    ROICropper(test_params).make_square_crops()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="config file path", type=str, required=True)

    return parser.parse_args()


if __name__ == '__main__':
    log_util.config(__file__)
    logger = logging.getLogger(__name__)

    args = parse_arguments()
    cropper_cfg = io_utils.json_load(args.config)

    try:
        run(cropper_cfg)
    except Exception as err:
        logger.error(err, exc_info=True)
        raise err
