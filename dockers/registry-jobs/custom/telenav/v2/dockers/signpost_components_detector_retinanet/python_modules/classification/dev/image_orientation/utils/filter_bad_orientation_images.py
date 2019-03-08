import argparse
import logging
import os
import shutil

import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
from classification.scripts.constants import Column, INVALID_IMG_PRED
from classification.scripts.prediction.folder_predictor import FolderPredictor


class BadOrientationImageFilter():
    IS_BAD_ORIENTATION_COL = "is_bad_orientation"

    def __init__(self, ftp_bundle_path, input_folder, output_folder):
        self.ftp_bundle_path = ftp_bundle_path
        self.input_folder = input_folder
        self.output_folder = output_folder
        io_utils.create_folder(self.output_folder)
        self.__logger = logging.getLogger(__name__)

    def __full_path(self, name):
        return

    def __is_bad_orientation(self, pred):
        return pred not in ["up", INVALID_IMG_PRED]

    def __move_bad_orientation_images_from_folder(self, preds_df):
        self.__logger.info("Moving images...")

        preds_df.loc[:, self.IS_BAD_ORIENTATION_COL] = preds_df.loc[:, Column.PRED_CLASS_COL] \
            .apply(lambda p: self.__is_bad_orientation(p))

        to_be_moved_df = preds_df[preds_df[self.IS_BAD_ORIENTATION_COL]]

        for img_name, row in to_be_moved_df.iterrows():
            source_img_path = os.path.join(self.input_folder, img_name)
            output_img_path = os.path.join(self.output_folder, img_name)

            self.__logger.info("Img {} has detected rotation of {}. Will be moved to output folder" \
                               .format(img_name, row[Column.PRED_CLASS_COL].upper(), output_img_path))

            shutil.move(source_img_path, output_img_path)

    def __build_predictor(self):
        return FolderPredictor(self.ftp_bundle_path)

    def filter_bad_orietation_images(self):
        predictor = self.__build_predictor()

        self.__logger.info("Predicting...")
        prediction_df = predictor.compute_prediction(self.input_folder)

        self.__move_bad_orientation_images_from_folder(prediction_df)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_folder", help="source images folder", type=str, required=True)
    parser.add_argument("-o", "--output_folder",
                        help="folder where the wrong orientation images will be moved", type=str, required=True)
    parser.add_argument("-w", "--ftp_bundle_path", help="ftp path to model weights", type=str, required=True)

    return parser.parse_args()


if __name__ == '__main__':

    log_util.config(__file__)
    logger = logging.getLogger(__name__)

    args = parse_arguments()
    try:
        image_filter = BadOrientationImageFilter(args.ftp_bundle_path, args.input_folder, args.output_folder)
        image_filter.filter_bad_orietation_images()
    except Exception as err:
        logger.error(err, exc_info=True)
        raise err
