import argparse
import logging
import os
import re
import shutil

import pandas as pd
from sklearn.utils import shuffle
from tqdm import tqdm

from apollo_python_common import log_util, io_utils
from classification.run.dataset_creator import DatasetCreator
from classification.scripts.constants import Column
from classification.scripts.signs_facing_classifier.constants import SignFacingLabel
import classification.scripts.utils as utils
import classification.scripts.signs_facing_classifier.constants as sf_constants


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--train_ds_cfg_json", help="path to json containing train dataset params",
                        type=str, required=True)
    parser.add_argument('-g', "--train_ds_gen_cfg_json",
                        help="path to json containing the sign facing train dataset generator params", type=str,
                        required=True)
    parser.add_argument('-ct', "--test_ds_cfg_json", help="path to json containing test dataset params",
                        type=str, required=True)
    parser.add_argument('-gt', "--test_ds_gen_cfg_json",
                        help="path to json containing the sign facing test dataset generator params", type=str,
                        required=True)

    return parser.parse_args()


class SignFacingDatasetGenerator:
    # generator config input paramenters
    BASE_INPUT_DIR = "base_input_dir"
    BASE_OUTPUT_DIR = "base_output_dir"
    FILTER_OUTPUT_DIR = "filter_output_dir"
    AREA_FILTER = "area"
    CLASS_THRESHOLD = "threshold"

    # directories corresponding to the three classes we want to handle
    FRONT_DIR = SignFacingLabel.CLS_FRONT
    LEFT_DIR = SignFacingLabel.CLS_LEFT
    RIGHT_DIR = SignFacingLabel.CLS_RIGHT

    FILENAME_SPLIT_REGEX = "_[0-9]+_[0-9]+_[0-9]+_[0-9]+_[A-Z]"
    IMAGE_EXTENSION = ".jpg"

    IMAGE_DF_COL = "image"
    TL_ROW_DF_COL = "tl_row"
    TL_COL_DF_COL = "tl_col"
    BR_ROW_DF_COL = "br_row"
    BR_COL_DF_COL = "br_col"
    ROI_AREA_DF_COL = "roi_area"
    ROI_CLASS_DF_COL = "roi_class"
    ROI_ORIENTATION_DF_COL = "roi_orientation"
    COUNT_DF_COL = "count"
    INDEX_DF_COL = "index"

    def __init__(self, config, filtered_classes_df=None):
        self.input_dir = config[self.BASE_INPUT_DIR]
        self.output_dir = config[self.BASE_OUTPUT_DIR]
        self.filter_output_dir = config[self.FILTER_OUTPUT_DIR]
        self.area = config[self.AREA_FILTER]
        self.threshold = config[self.CLASS_THRESHOLD]
        self.filtered_classes_df = filtered_classes_df
        self._create_output_paths()

    def _create_output_paths(self):
        """ Creates all the output directories, if they don't already exist. """

        io_utils.create_folder(self.filter_output_dir)
        io_utils.create_folder(os.path.join(self.output_dir, self.FRONT_DIR))
        io_utils.create_folder(os.path.join(self.output_dir, self.LEFT_DIR))
        io_utils.create_folder(os.path.join(self.output_dir, self.RIGHT_DIR))

    def _split_img_name(self, image_name, input_dir, orientation):
        """ Splits the given image name in it's components, sets the originating image name, and returns the result. """

        img = image_name.replace(os.path.join(input_dir, orientation) + '/', '')
        img = img.replace(self.IMAGE_EXTENSION, '')

        coords = re.search(self.FILENAME_SPLIT_REGEX, img).group(0)
        coords_split = coords.split('_')

        name_and_class = re.split(self.FILENAME_SPLIT_REGEX, img)

        raw_img_name = name_and_class[0]
        roi_class = coords_split[5] + name_and_class[1]

        # coords = row['tl_col'], row['tl_row'], row['br_col'], row['br_row']
        return raw_img_name, coords_split[2], coords_split[1], coords_split[4], coords_split[3], roi_class

    def _make_roi_dataframe(self, img_list, orientation):
        """ Given a list of image names, returns a corresponding dataframe which contains the pieces from the
         image name. """

        data_dict = {self.IMAGE_DF_COL: [], self.TL_ROW_DF_COL: [], self.TL_COL_DF_COL: [], self.BR_ROW_DF_COL: [],
                     self.BR_COL_DF_COL: [], self.ROI_AREA_DF_COL: [], self.ROI_CLASS_DF_COL: [],
                     self.ROI_ORIENTATION_DF_COL: []}

        for img in img_list:
            pieces = self._split_img_name(img, self.input_dir, orientation)

            data_dict[self.IMAGE_DF_COL].append(pieces[0])
            data_dict[self.TL_ROW_DF_COL].append(pieces[1])
            data_dict[self.TL_COL_DF_COL].append(pieces[2])
            data_dict[self.BR_ROW_DF_COL].append(pieces[3])
            data_dict[self.BR_COL_DF_COL].append(pieces[4])
            data_dict[self.ROI_AREA_DF_COL].append(
                utils.compute_roi_area(int(pieces[4]), int(pieces[2]), int(pieces[3]), int(pieces[1])))
            data_dict[self.ROI_CLASS_DF_COL].append(pieces[5])
            data_dict[self.ROI_ORIENTATION_DF_COL].append(orientation)

        return pd.DataFrame(data_dict)

    def _load_imgs_in_dataframes(self):
        """ Loads the original tagged images from their respective front, left or right directories and creates a
        dataframe for each class, referencing the image from which the ROI originated. These dataframes are returned
        to the caller of the function. """

        front_imgs = io_utils.get_images_from_folder(os.path.join(self.input_dir, self.FRONT_DIR))
        left_imgs = io_utils.get_images_from_folder(os.path.join(self.input_dir, self.LEFT_DIR))
        right_imgs = io_utils.get_images_from_folder(os.path.join(self.input_dir, self.RIGHT_DIR))

        front_df = self._make_roi_dataframe(front_imgs, SignFacingLabel.CLS_RIGHT)
        left_df = self._make_roi_dataframe(left_imgs, SignFacingLabel.CLS_LEFT)
        right_df = self._make_roi_dataframe(right_imgs, SignFacingLabel.CLS_RIGHT)

        return front_df, left_df, right_df

    def _apply_class_threshold(self, dataframe):
        """
        Applies the threshold config value to filter out the ROI labels which have less than the threshold instances
        from the given dataframe.
        :param dataframe: the data frame to be filtered
        :return: the filtered data frame
        """

        print('apply class threshold: ', self.threshold)
        filtered_df = pd.DataFrame(dataframe[self.ROI_CLASS_DF_COL].value_counts()).reset_index()
        filtered_df = filtered_df.rename(columns={self.ROI_CLASS_DF_COL: self.COUNT_DF_COL,
                                                  self.INDEX_DF_COL: self.ROI_CLASS_DF_COL})
        filtered_df = filtered_df[filtered_df[self.COUNT_DF_COL] > self.threshold]

        joined_df = pd.merge(dataframe, filtered_df, how='inner', left_on=[self.ROI_CLASS_DF_COL],
                             right_on=[self.ROI_CLASS_DF_COL])

        return joined_df

    def _filter_with_threshold(self, front_df, left_df, right_df):
        """ Filters out the ROI labels that have less instances than the config threshold from the front, left and right
         data frames. Also sets the filtered_classes_df with these ROI labels, so it can be re-used."""

        not_front_df = pd.concat([left_df, right_df])
        logger.info('----not front df unfiltered----')
        logger.info(not_front_df[self.ROI_CLASS_DF_COL].value_counts())

        not_front_df = self._apply_class_threshold(not_front_df)
        logger.info('----not front df filtered----')
        logger.info(not_front_df[self.ROI_CLASS_DF_COL].value_counts())

        self.filtered_classes_df = pd.DataFrame({self.ROI_CLASS_DF_COL: not_front_df[self.ROI_CLASS_DF_COL].unique()})
        ft_front_df, ft_left_df, ft_right_df = self._filter_with_class_list(front_df, left_df, right_df)

        return ft_front_df, ft_left_df, ft_right_df

    def _filter_with_class_list(self, front_df, left_df, right_df):
        """ Filters the front, left and right dataframes using the dataframe with the filtered classes with which the
        generator was initialized, if not None. """

        ft_front_df = pd.merge(front_df, self.filtered_classes_df, how='inner', left_on=[self.ROI_CLASS_DF_COL],
                               right_on=[self.ROI_CLASS_DF_COL])
        ft_left_df = pd.merge(left_df, self.filtered_classes_df, how='inner', left_on=[self.ROI_CLASS_DF_COL],
                              right_on=[self.ROI_CLASS_DF_COL])
        ft_right_df = pd.merge(right_df, self.filtered_classes_df, how='inner', left_on=[self.ROI_CLASS_DF_COL],
                               right_on=[self.ROI_CLASS_DF_COL])

        return ft_front_df, ft_left_df, ft_right_df

    def _filter_by_area(self, dataframe):
        """ Filter the dataframe based on the area column, removing items smaller than the given area argument. """

        filtered_df = dataframe[dataframe[self.ROI_AREA_DF_COL] >= self.area]

        return filtered_df

    def _apply_filters(self, front_df, left_df, right_df):
        """Applies the area and threshold filters over the front, left and right data frames. """

        logger.info('applying filter with threshold {}...'.format(self.threshold))
        if self.filtered_classes_df is None:
            ft_front_df, ft_left_df, ft_right_df = self._filter_with_threshold(front_df, left_df, right_df)
        else:
            ft_front_df, ft_left_df, ft_right_df = self._filter_with_class_list(front_df, left_df, right_df)

        logger.info('applying filter with area {}...'.format(self.area))
        filtered_front_df = self._filter_by_area(ft_front_df)
        filtered_left_df = self._filter_by_area(ft_left_df)
        filtered_right_df = self._filter_by_area(ft_right_df)

        return filtered_front_df, filtered_left_df, filtered_right_df

    def _copy_images(self, filtered_df, orientation):
        """ Copies the filtered images to the destination folder corresponding to the given orientation. """

        logger.info('copying {} images...'.format(orientation))
        logger.info('filtered images count is {}'.format(filtered_df.shape[0]))

        count = 0
        source_path = os.path.join(self.input_dir, orientation)
        for idx, row in tqdm(filtered_df.iterrows()):
            src_img = '{}_{}_{}_{}_{}_{}.jpg'.format(row[self.IMAGE_DF_COL], row[self.TL_COL_DF_COL],
                                                     row[self.TL_ROW_DF_COL], row[self.BR_COL_DF_COL],
                                                     row[self.BR_ROW_DF_COL], row[self.ROI_CLASS_DF_COL])
            shutil.copy2(os.path.join(source_path, src_img), os.path.join(self.output_dir, orientation))
            count += 1

        logger.info('finished copying {} files.'.format(count))

    def save_filter_data(self):
        """ Saves the filter values used by the generator to generate the data set. """

        filter_values = {'keep_classes': self.filtered_classes_df[self.ROI_CLASS_DF_COL].unique().tolist(),
                         'threshold_value': self.threshold, 'area_value': self.area}

        io_utils.json_dump(filter_values, os.path.join(self.filter_output_dir, sf_constants.MODEL_FILTERS))

    def generate_dataset(self):
        """ Runs the dataset generator on the given input dir of images to apply the filters and create the output
         dataset."""

        logger.info("generate dataset from input dir {} area {} and threshold {}".format(self.input_dir, self.area,
                                                                                         self.threshold))

        front_df, left_df, right_df = self._load_imgs_in_dataframes()
        ft_front_df, ft_left_df, ft_right_df = self._apply_filters(front_df, left_df, right_df)

        logger.info('front image source count is {}'.format(front_df.shape[0]))
        self._copy_images(ft_front_df, self.FRONT_DIR)

        logger.info('left image source count is {}'.format(left_df.shape[0]))
        self._copy_images(ft_left_df, self.LEFT_DIR)

        logger.info('right image source count is {}'.format(right_df.shape[0]))
        self._copy_images(ft_right_df, self.RIGHT_DIR)


class SignFacingDatasetCreator(DatasetCreator):
    def __init__(self, ds_config):
        super().__init__(ds_config)

    def train_test_split(self, data_df):
        train_list = []
        test_list = []

        for label_class, grouped_df in data_df.groupby(Column.LABEL_CLASS_COL):
            nr_train = int(self.TRAIN_TEST_SPLIT_PERCENTAGE * len(grouped_df))
            grouped_df = shuffle(grouped_df)

            train_df = grouped_df[:nr_train]
            test_df = grouped_df[nr_train:]

            train_list.append(train_df)
            test_list.append(test_df)

        train_df = pd.concat(train_list)
        test_df = pd.concat(test_list)

        return train_df, test_df


if __name__ == '__main__':

    log_util.config(__file__)
    logger = logging.getLogger(__name__)

    args = parse_arguments()

    train_ds_cfg = io_utils.json_load(args.train_ds_cfg_json)
    test_ds_cfg = io_utils.json_load(args.test_ds_cfg_json)
    train_ds_gen_cfg = io_utils.json_load(args.train_ds_gen_cfg_json)
    test_ds_gen_cfg = io_utils.json_load(args.test_ds_gen_cfg_json)

    try:
        # run the dataset generator on both train and test images for filtering the sign facing classifier images
        train_generator = SignFacingDatasetGenerator(train_ds_gen_cfg)
        train_generator.generate_dataset()
        train_generator.save_filter_data()
        SignFacingDatasetGenerator(test_ds_gen_cfg, train_generator.filtered_classes_df).generate_dataset()

        # run the dataset creator from the classification components on both train and test images resulting from the
        # filtering process above
        SignFacingDatasetCreator(train_ds_cfg).create_dataset()
        SignFacingDatasetCreator(test_ds_cfg).create_dataset()
    except Exception as err:
        logger.error(err, exc_info=True)
        raise err
