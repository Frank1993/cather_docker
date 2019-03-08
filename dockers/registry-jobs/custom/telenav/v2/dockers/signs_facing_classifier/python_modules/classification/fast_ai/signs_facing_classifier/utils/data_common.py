import os
import re

import pandas as pd

from apollo_python_common import io_utils
from apollo_python_common.rectangle import Rectangle
from classification.fast_ai.signs_facing_classifier.utils.constants import RoiDfColumn
from classification.fast_ai.signs_facing_classifier.utils.constants import SignFacingLabel
import classification.fast_ai.signs_facing_classifier.utils.constants as const


def split_img_name(image_name):
    """ Splits the given image name in it's components, sets the originating image name, and returns the result. """
    img = os.path.basename(image_name).split('.')[0]

    coords = re.search(const.FILENAME_SPLIT_REGEX, img).group(0)
    coords_split = coords.split('_')

    name_and_class = re.split(const.FILENAME_SPLIT_REGEX, img)

    raw_img_name = name_and_class[0]
    roi_class = coords_split[5] + name_and_class[1]

    return pd.Series({RoiDfColumn.IMG_NAME_COL: raw_img_name, RoiDfColumn.TL_ROW_COL: int(coords_split[2]),
                      RoiDfColumn.TL_COL_COL: int(coords_split[1]),
                      RoiDfColumn.BR_ROW_COL: int(coords_split[4]), RoiDfColumn.BR_COL_COL: int(coords_split[3]),
                      RoiDfColumn.ROI_CLASS_COL: roi_class})


def get_orig_images(new_dir, old_dir):
    old_imgs = set([os.path.basename(file_path) for file_path in io_utils.get_images_from_folder(old_dir)])
    new_imgs = set([os.path.basename(file_path) for file_path in io_utils.get_images_from_folder(new_dir)])

    return new_imgs, old_imgs


def get_orig_img_path(img_name, new_imgs, old_imgs, new_dir, old_dir):
    """ Given an original image name without extension, it returns the full path (including directories) of
        the image. """
    jpg = '{}.jpg'.format(img_name)
    jpeg = '{}.jpeg'.format(img_name)

    if jpg in new_imgs:
        path = os.path.join(new_dir, jpg)
    elif jpeg in new_imgs:
        path = os.path.join(new_dir, jpeg)
    elif jpg in old_imgs:
        path = os.path.join(old_dir, jpg)
    else:
        path = os.path.join(old_dir, jpeg)

    return path


def make_roi_dataframe(img_list, orientation, with_area=True):
    """ Given a list of image names, returns a corresponding dataframe which contains the pieces from the
     image name. """
    roi_df = pd.DataFrame(img_list, columns=[const.TMP_DF_COL])
    roi_df = pd.concat([roi_df, roi_df[const.TMP_DF_COL].apply(split_img_name)], axis=1)
    roi_df = roi_df.drop([const.TMP_DF_COL], axis=1)
    roi_df[RoiDfColumn.ORIENTATION_COL] = orientation

    if with_area:
        roi_df[RoiDfColumn.ROI_AREA_COL] = roi_df.apply(
            lambda row: Rectangle(row[RoiDfColumn.TL_COL_COL], row[RoiDfColumn.TL_ROW_COL], row[RoiDfColumn.BR_COL_COL],
                                  row[RoiDfColumn.BR_ROW_COL]).area(), axis=1)

    return roi_df


def load_imgs_in_dataframes(input_dir, with_area=True):
    """ Loads the original tagged images from their respective front, left or right directories and creates a
    dataframe for each class, referencing the image from which the ROI originated. These dataframes are returned
    to the caller of the function. """

    front_imgs = io_utils.get_images_from_folder(os.path.join(input_dir, SignFacingLabel.FRONT))
    left_imgs = io_utils.get_images_from_folder(os.path.join(input_dir, SignFacingLabel.LEFT))
    right_imgs = io_utils.get_images_from_folder(os.path.join(input_dir, SignFacingLabel.RIGHT))

    front_df = make_roi_dataframe(front_imgs, SignFacingLabel.FRONT, with_area)
    left_df = make_roi_dataframe(left_imgs, SignFacingLabel.LEFT, with_area)
    right_df = make_roi_dataframe(right_imgs, SignFacingLabel.RIGHT, with_area)

    return front_df, left_df, right_df
