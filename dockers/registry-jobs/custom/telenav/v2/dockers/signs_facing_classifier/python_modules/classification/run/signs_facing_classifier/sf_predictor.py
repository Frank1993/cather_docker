import argparse
import logging
import os
import shutil

from apollo_python_common import log_util, io_utils, ftp_utils
from classification.scripts.signs_facing_classifier import postprocess
from classification.scripts.constants import PredictorCfgParams
from classification.scripts.prediction.folder_predictor import FolderPredictor


class SignFacingPredictor:
    APPLY_BEST_THRESHOLDS_PARAM = "apply_best_thresholds"
    PREDICTION_FILE = "model_predictions.csv"
    MODEL_BUNDLE_PATH = "./model_bundle"
    BEST_THREHSHOLDS_PATH = os.path.join(MODEL_BUNDLE_PATH, "model_best_thresholds.json")

    def __init__(self, predict_cfg):
        self.predict_config = predict_cfg
        self.folder_predictor = self._build_folder_predictor()

    def _build_folder_predictor(self):
        return FolderPredictor(self.predict_config[PredictorCfgParams.FTP_BUNDLE_PATH_PARAM],
                               self.predict_config[PredictorCfgParams.NR_IMGS_PARAM],
                               with_img=False)

    def _compute_predictions(self):
        return self.folder_predictor.compute_prediction(self.predict_config[PredictorCfgParams.INPUT_FOLDER_PARAM])

    def _save_results(self, pred_df):
        pred_df = postprocess.get_pred_confidence_df(pred_df, self.folder_predictor.params)
        logger.info("saving predicted results {}...".format(os.getcwd()))

        if self.predict_config[self.APPLY_BEST_THRESHOLDS_PARAM]:
            ftp_utils.copy_zip_and_extract(self.predict_config[PredictorCfgParams.FTP_BUNDLE_PATH_PARAM],
                                           self.MODEL_BUNDLE_PATH)
            lt, rt = postprocess.load_best_thresholds_json(self.BEST_THREHSHOLDS_PATH)
            logger.info("applying best thresholds - lt: {}, rt: {}".format(lt, rt))
            pred_df = postprocess.apply_thresholds(pred_df, lt, rt)
            shutil.rmtree(self.MODEL_BUNDLE_PATH)

        pred_df.to_csv(os.path.join(self.predict_config[PredictorCfgParams.OUTPUT_FOLDER_PARAM], self.PREDICTION_FILE))

    def predict(self):
        pred_df = self._compute_predictions()
        self._save_results(pred_df)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--predict_config_json", help="path to json containing predict params",
                        type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':

    log_util.config(__file__)
    logger = logging.getLogger(__name__)

    args = parse_arguments()
    predict_config = io_utils.json_load(args.predict_config_json)

    try:
        SignFacingPredictor(predict_config).predict()
    except Exception as err:
        logger.error(err, exc_info=True)
        raise err