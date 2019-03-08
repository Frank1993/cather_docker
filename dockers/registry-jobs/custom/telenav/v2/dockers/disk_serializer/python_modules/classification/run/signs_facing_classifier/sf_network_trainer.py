import argparse
import logging
import os
import shutil

from keras.callbacks import ModelCheckpoint, CSVLogger

import classification.scripts.signs_facing_classifier.postprocess as postprocess
import classification.scripts.signs_facing_classifier.constants as sf_constants
import apollo_python_common.ftp_utils as ftp
from apollo_python_common import log_util, io_utils
from classification.scripts import utils, network, generators
from classification.scripts.constants import NetworkCfgParams
from classification.scripts.constants import Column


class SignFacingNetworkTrainer:
    MODEL_FILTERS_PATH = "model_filters_path"
    COMPUTE_BEST_THREHSOLDS = "compute_best_thresholds"
    VALIDATED_PARAMS_PATH = "validated_params_path"
    FTP_BUNDLE_PATH = "./ftp_bundle"
    MODEL_PATH = "./sign_facing_best.h5"
    JSON_THRESHOLDS_PATH = os.path.join(FTP_BUNDLE_PATH, "model_best_thresholds.json")
    CSV_THREHSOLDS_PATH = os.path.join(FTP_BUNDLE_PATH, "model_best_thresholds.csv")

    def __init__(self, train_cfg):
        self.train_config = train_cfg
        self.model_params, self.validated_params = self._load_params()

    def _load_params(self):
        model_params = utils.json_load_classif_params(self.train_config[NetworkCfgParams.MODEL_PARAMS_PATH_PARAM])
        validated_params = utils.json_load_classif_params(self.train_config[self.VALIDATED_PARAMS_PATH])

        return model_params, validated_params

    def _build_model(self):
        network_config = self.train_config[NetworkCfgParams.NETWORK_CONFIG_PARAM]
        input_shape = network.get_conv_model(self.model_params.conv_layer_name,
                                             self.model_params.img_size).output_shape[1:]

        model = network.get_ssd_conv_model(input_shape,
                                           network_config[NetworkCfgParams.NR_CONV_BLOCKS_PARAM],
                                           network_config[NetworkCfgParams.NR_FILTERS_PARAM],
                                           network_config[NetworkCfgParams.DROPOUT_LEVEL_PARAM],
                                           len(self.model_params.classes))
        return model

    def _build_data_generators(self):
        df_path_list = [self.model_params.train_df_path, self.model_params.test_df_path]
        conv_path_list = [self.model_params.train_conv_path, self.model_params.test_conv_path]

        validated_df_path_list = [self.validated_params.train_df_path, self.validated_params.test_df_path]
        validated_conv_path_list = [self.validated_params.train_conv_path, self.validated_params.test_conv_path]

        train_generator = generators.ClassifGenerator(df_path_list, conv_path_list)
        test_generator = generators.ClassifGenerator(validated_df_path_list, validated_conv_path_list)

        return train_generator, test_generator

    def _train_model(self, model, train_generator, test_generator):
        nr_train_data = len(os.listdir(self.model_params.train_df_path))
        nr_test_data = len(os.listdir(self.model_params.test_df_path))
        nr_val_train_data = len(os.listdir(self.validated_params.train_df_path))
        nr_val_test_data = len(os.listdir(self.validated_params.test_df_path))

        model.fit_generator(train_generator,
                            steps_per_epoch=nr_train_data + nr_test_data,
                            epochs=self.train_config[NetworkCfgParams.NR_EPOCHS_PARAM],
                            validation_data=test_generator,
                            validation_steps=nr_val_train_data + nr_val_test_data,
                            callbacks=[CSVLogger(self.train_config[NetworkCfgParams.TRAIN_PROGRESS_CSV_PARAM]),
                                       ModelCheckpoint(self.MODEL_PATH, monitor='val_acc', verbose=1,
                                                       save_best_only=True,
                                                       mode='max')
                                       ],
                            workers=10)
        model.load_weights(self.MODEL_PATH)  # load the best weights

        return model

    def _predict_on_testset(self, model):
        test_df_path_list = [self.validated_params.train_df_path, self.validated_params.test_df_path]
        test_img_path_list = [self.validated_params.train_img_path, self.validated_params.test_img_path]
        test_conv_path_list = [self.validated_params.train_conv_path, self.validated_params.test_conv_path]

        pred_data_df = network.make_prediction_on_dataset(test_df_path_list,
                                                          test_img_path_list,
                                                          test_conv_path_list,
                                                          model,
                                                          nr_batches=None,
                                                          with_img=True)
        pred_data_df.loc[:, Column.PRED_CLASS_COL] = pred_data_df.loc[:, Column.PRED_COL].apply(
            lambda pred: utils.label2text(pred, self.model_params.classIndex_2_class))

        return pred_data_df

    def _save_best_thresholds(self, thresholds_df):
        logger.info("input threhsolds_df \n", thresholds_df)

        lt = float(thresholds_df.at[0, sf_constants.THRESHOLD_NAME_LEFT])
        rt = float(thresholds_df.at[0, sf_constants.THRESHOLD_NAME_RIGHT])
        score = int(thresholds_df.at[0, sf_constants.MODEL_SCORE])

        best_thresholds = {sf_constants.THRESHOLD_NAME_LEFT: lt, sf_constants.THRESHOLD_NAME_RIGHT: rt,
                           sf_constants.MODEL_SCORE: score}
        logger.info("lt: {}, rt: {}, score: {} - best thresholds: {}".format(lt, rt, score, best_thresholds))
        io_utils.json_dump(best_thresholds, self.JSON_THRESHOLDS_PATH)
        thresholds_df.to_csv(self.CSV_THREHSOLDS_PATH)

    def _compute_best_thresholds(self, model):
        logger.info("Computing best thresholds: {}".format(self.train_config[self.COMPUTE_BEST_THREHSOLDS]))
        if self.train_config[self.COMPUTE_BEST_THREHSOLDS]:
            pred_df = self._predict_on_testset(model)
            thresholds_df = postprocess.compute_best_thresholds(
                postprocess.get_pred_confidence_df(pred_df, self.model_params))
            self._save_best_thresholds(thresholds_df)

    def _save_bundle_to_ftp(self, model):
        logger.info("Saving model data {} to FTP {}...".format(self.FTP_BUNDLE_PATH,
                                                               self.train_config[NetworkCfgParams.FTP_OUTPUT_PARAM]))

        shutil.copy2(self.train_config[self.MODEL_FILTERS_PATH], self.FTP_BUNDLE_PATH)  # copy the model filters
        network.save_model_bundle(self.FTP_BUNDLE_PATH, model, self.model_params)
        ftp.dir_copy_local_to_ftp(ftp.FTP_SERVER, ftp.FTP_USER_NAME, ftp.FTP_PASSWORD,
                                  self.FTP_BUNDLE_PATH,
                                  self.train_config[NetworkCfgParams.FTP_OUTPUT_PARAM])

    def _clean_workspace(self):
        shutil.rmtree(self.FTP_BUNDLE_PATH)
        os.remove(self.MODEL_PATH)

    def train_network(self):
        train_generator, test_generator = self._build_data_generators()

        model = self._build_model()
        model = self._train_model(model, train_generator, test_generator)

        self._compute_best_thresholds(model)
        self._save_bundle_to_ftp(model)
        self._clean_workspace()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--train_config_json", help="path to json containing training params",
                        type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':

    log_util.config(__file__)
    logger = logging.getLogger(__name__)

    args = parse_arguments()
    train_config = io_utils.json_load(args.train_config_json)

    try:
        SignFacingNetworkTrainer(train_config).train_network()
    except Exception as err:
        logger.error(err, exc_info=True)
        raise err
