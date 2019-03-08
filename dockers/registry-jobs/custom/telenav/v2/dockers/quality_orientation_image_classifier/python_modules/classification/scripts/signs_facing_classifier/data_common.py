import os
import re

import pandas as pd
from sklearn.utils import shuffle

from apollo_python_common import io_utils
from classification.scripts.signs_facing_classifier.constants import SignFacingLabel
from classification.scripts.signs_facing_classifier.constants import RoiDfColumn
import classification.scripts.utils as utils
import classification.scripts.signs_facing_classifier.constants as const


def split_img_name(image_name):
    """ Splits the given image name in it's components, sets the originating image name, and returns the result. """
    img = os.path.basename(image_name).split('.')[0]

    coords = re.search(const.FILENAME_SPLIT_REGEX, img).group(0)
    coords_split = coords.split('_')

    name_and_class = re.split(const.FILENAME_SPLIT_REGEX, img)

    raw_img_name = name_and_class[0]
    roi_class = coords_split[5] + name_and_class[1]

    return raw_img_name, coords_split[2], coords_split[1], coords_split[4], coords_split[3], roi_class


def make_roi_dataframe(img_list, orientation, with_area=True):
    """ Given a list of image names, returns a corresponding dataframe which contains the pieces from the
     image name. """
    data_dict = {RoiDfColumn.IMG_NAME_COL: [], RoiDfColumn.TL_ROW_COL: [], RoiDfColumn.TL_COL_COL: [],
                 RoiDfColumn.BR_ROW_COL: [], RoiDfColumn.BR_COL_COL: [], RoiDfColumn.ROI_CLASS_COL: []}

    for img in img_list:
        pieces = split_img_name(img)

        data_dict[RoiDfColumn.IMG_NAME_COL].append(pieces[0])
        data_dict[RoiDfColumn.TL_ROW_COL].append(int(pieces[1]))
        data_dict[RoiDfColumn.TL_COL_COL].append(int(pieces[2]))
        data_dict[RoiDfColumn.BR_ROW_COL].append(int(pieces[3]))
        data_dict[RoiDfColumn.BR_COL_COL].append(int(pieces[4]))
        data_dict[RoiDfColumn.ROI_CLASS_COL].append(pieces[5])

    data_dict[RoiDfColumn.ORIENTATION_COL] = orientation

    roi_df = pd.DataFrame(data_dict)
    if with_area:
        roi_df[RoiDfColumn.ROI_AREA_COL] = roi_df.apply(lambda row: utils.compute_roi_area(row[RoiDfColumn.BR_COL_COL],
                                                                                           row[RoiDfColumn.TL_COL_COL],
                                                                                           row[RoiDfColumn.BR_ROW_COL],
                                                                                           row[RoiDfColumn.TL_ROW_COL]),
                                                        axis=1)

    return roi_df


def load_imgs_in_dataframes(input_dir, with_area=True):
    """ Loads the original tagged images from their respective front, left or right directories and creates a
    dataframe for each class, referencing the image from which the ROI originated. These dataframes are returned
    to the caller of the function. """

    front_imgs = io_utils.get_images_from_folder(os.path.join(input_dir, SignFacingLabel.CLS_FRONT))
    left_imgs = io_utils.get_images_from_folder(os.path.join(input_dir, SignFacingLabel.CLS_LEFT))
    right_imgs = io_utils.get_images_from_folder(os.path.join(input_dir, SignFacingLabel.CLS_RIGHT))

    front_df = make_roi_dataframe(front_imgs, SignFacingLabel.CLS_FRONT, with_area)
    left_df = make_roi_dataframe(left_imgs, SignFacingLabel.CLS_LEFT, with_area)
    right_df = make_roi_dataframe(right_imgs, SignFacingLabel.CLS_RIGHT, with_area)

    return front_df, left_df, right_df


def train_test_split_stratified(data_df, percentage, groupby):
    """ Given a dataframe containing a full set of metadata about ROI facing images, it splits it using the percentage
     argument in a stratified manner.
     :param groupby: """

    train_list = []
    test_list = []

    for label_class, grouped_df in data_df.groupby(groupby):
        nr_train = int(percentage * len(grouped_df))
        grouped_df = shuffle(grouped_df, random_state=0)

        train_df = grouped_df[:nr_train]
        test_df = grouped_df[nr_train:]

        train_list.append(train_df)
        test_list.append(test_df)

    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list)

    return train_df, test_df
