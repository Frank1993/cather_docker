import argparse
import logging
import os
import shutil

import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
from classification.scripts.constants import Column
from classification.scripts.prediction.folder_predictor import FolderPredictor


class BadQualityImageFilter():
    IS_BAD_QUALITY_COL = "is_bad_quality"

    def __init__(self, ftp_bundle_path, input_folder, output_folder):
        self.ftp_bundle_path = ftp_bundle_path
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.logger = logging.getLogger(__name__)
        io_utils.create_folder(self.output_folder)

    def __is_bad_quality(self, row):
        return row[Column.PRED_CLASS_COL] == "bad" and row[Column.PRED_CONF_COL] > 0.98

    def __move_bad_quality_images_from_folder(self, preds_df):
        self.logger.info("Moving images...")

        preds_df.loc[:, self.IS_BAD_QUALITY_COL] = preds_df.apply(lambda r: self.__is_bad_quality(r), axis=1)

        to_be_moved_df = preds_df[preds_df[self.IS_BAD_QUALITY_COL]]

        for img_name, row in to_be_moved_df.iterrows():
            source_img_path = os.path.join(self.input_folder, img_name)
            output_img_path = os.path.join(self.output_folder, img_name)

            self.logger.info("Detected image {} with bad quality. Will be moved to output folder".format(img_name,
                                                                                                         output_img_path))

            shutil.move(source_img_path, output_img_path)

    def __build_predictor(self):
        return FolderPredictor(self.ftp_bundle_path)

    def filter_bad_quality_images(self):
        predictor = self.__build_predictor()

        self.logger.info("Predicting...")
        prediction_df = predictor.compute_prediction(self.input_folder)

        self.__move_bad_quality_images_from_folder(prediction_df)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_folder", help="source images folder", type=str, required=True)
    parser.add_argument("-o", "--output_folder",
                        help="folder where the wrong quality images will be moved", type=str, required=True)
    parser.add_argument("-w", "--ftp_bundle_path", help="ftp path to model weights", type=str, required=True)

    return parser.parse_args()

if __name__ == '__main__':

    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    args = parse_arguments()
    try:
        image_filter = BadQualityImageFilter(args.ftp_bundle_path, args.input_folder, args.output_folder)
        image_filter.filter_bad_quality_images()
    except Exception as err:
        logger.error(err, exc_info=True)
        raise err
