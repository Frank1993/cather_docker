import argparse
import logging
import os

import pandas as pd

from fastai.vision import *

from apollo_python_common import log_util, io_utils
from classification.fast_ai.common import predict
from classification.fast_ai.signs_facing_classifier.utils import data_common, model_common
from classification.fast_ai.signs_facing_classifier.utils.constants import SignFacingLabel
import classification.fast_ai.signs_facing_classifier.utils.constants as const
from classification.fast_ai.signs_facing_classifier.utils.constants import PredDfColumn
from classification.fast_ai.signs_facing_classifier.utils.constants import RoiDfColumn


class SignsFacingPredictor:
    """ Uses a FastAI learner to predict over a folder of images. """

    def __init__(self, params):
        self.params = params
        self.classes = io_utils.json_load(os.path.join(self.params.model_dir, self.params.label_list_file)).label_list
        logger.info("labels: {}".format(self.classes))

        self.learner = predict.get_inference_learner(self.params, self.classes,
                                                     const.MODEL_DICT[self.params.backbone_model])

    def _load_to_df(self):
        img_df = pd.concat(list(data_common.load_imgs_in_dataframes(self.params.imgs_dir, with_area=False)))
        logger.info("size of set to predict: {}".format(len(img_df)))

        img_df[RoiDfColumn.IMG_PATH] = img_df.apply(lambda row: model_common.get_img_path(row, self.params.imgs_dir),
                                                    axis=1)

        return img_df

    def _predict_on_image(self, row):
        """ Run prediction on the given image from a dataset row, using the given model. """
        img = open_image(row['img_path']).resize((3, self.params.image_size, self.params.image_size))
        pred = self.learner.predict(img)

        return pred

    def _adjust_preds(self, pred_df):
        """ Adjust the predictions returned by the FastAI library to be separate confidence columns in the
         pred dataframe. """

        pred_df[PredDfColumn.PRED_CLASS_COL] = pred_df[PredDfColumn.PRED_COL].apply(lambda pred: str(pred[0]))
        pred_df[PredDfColumn.CONF_FRONT_COL] = pred_df[PredDfColumn.PRED_COL].apply(
            lambda pred: pred[2][self.classes.index(SignFacingLabel.FRONT)].item())
        pred_df[PredDfColumn.CONF_LEFT_COL] = pred_df[PredDfColumn.PRED_COL].apply(
            lambda pred: pred[2][self.classes.index(SignFacingLabel.LEFT)].item())
        pred_df[PredDfColumn.CONF_RIGHT_COL] = pred_df[PredDfColumn.PRED_COL].apply(
            lambda pred: pred[2][self.classes.index(SignFacingLabel.RIGHT)].item())

        return pred_df.drop([PredDfColumn.PRED_COL], axis=1)

    def predict(self):
        pred_df = self._load_to_df()
        pred_df[PredDfColumn.PRED_COL] = pred_df.apply(lambda row: self._predict_on_image(row), axis=1)

        pred_df = self._adjust_preds(pred_df)

        logger.info("model score: {})".format(model_common.compute_model_score(pred_df)))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--predict_cfg_json", help="path to json containing training params",
                        type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    log_util.config(__file__)
    logger = logging.getLogger(__name__)

    args = parse_arguments()
    predict_config = io_utils.json_load(args.predict_cfg_json)

    try:
        predictor = SignsFacingPredictor(predict_config)
        predictor.predict()
    except Exception as err:
        logger.error(err, exc_info=True)
        raise err
