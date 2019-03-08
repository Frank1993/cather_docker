import argparse
import logging
import os

from tqdm import tqdm

from roadsense.scripts.aggregator import HazardAggregator
from roadsense.scripts.hazard_detector import HazardDetector
from roadsense.scripts.dataset_preprocessor import DatasetPreprocessor

from roadsense.scripts.config import ConfigParams as cp, HazardType as HazardType

import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util

tqdm.pandas()


class Trainer:
    def __init__(self, train_config):
        self.train_config = train_config
        self.__augment_train_config()
        self.preprocessor = DatasetPreprocessor(train_config)
        self.detector = HazardDetector(train_config)
        self.aggregator = HazardAggregator(train_config)

    def __get_index_dicts(self):
        index_2_class = dict(list(enumerate(self.train_config[cp.KEPT_HAZARDS] + [HazardType.CLEAR])))
        class_2_index = {v: k for k, v in index_2_class.items()}

        return index_2_class, class_2_index

    def __augment_train_config(self):
        index_2_class, class_2_index = self.__get_index_dicts()
        self.train_config[cp.INDEX_2_CLASS], self.train_config[cp.CLASS_2_INDEX] = index_2_class, class_2_index

    def __add_params_to_config(self, best_threshold, eps, min_samples):
        self.train_config[cp.CONF_THRESHOLD] = best_threshold
        self.train_config[cp.DBSCAN_EPS] = int(eps)
        self.train_config[cp.DBSCAN_MIN_SAMPLES] = int(min_samples)

    def __save_config(self, output_folder, best_threshold, eps, min_samples):
        self.__add_params_to_config(best_threshold, eps, min_samples)
        io_utils.json_dump(self.train_config, os.path.join(output_folder, "train_config.json"))

    def serialize_model_and_config(self, model, best_threshold, eps, min_samples):
        output_folder = self.train_config[cp.BUNDLE_PATH]
        io_utils.create_folder(output_folder)

        self.detector.save_model(model, output_folder)
        self.__save_config(output_folder, best_threshold, eps, min_samples)

    def train(self):
        X_train, y_train_ohe, X_test, y_test_ohe, test_df = \
            self.preprocessor.get_train_test_data()

        model = self.detector.train_model(X_train, y_train_ohe, X_test, y_test_ohe)

        y_pred_proba = self.detector.predict_with_model(model, X_test)
        best_conf_thresh = self.detector.get_best_conf_threshold(y_pred_proba, y_test_ohe)

        test_df = self.preprocessor.add_preds_to_df(test_df, y_pred_proba, best_conf_thresh)
        trip_test_df = self.preprocessor.filter_single_trip(test_df, "iPhone6")

        eps, min_samples = self.aggregator.get_best_clustering_params(trip_test_df)

        self.serialize_model_and_config(model, best_conf_thresh, eps, min_samples)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--config_json", help="path to config json", type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':

    log_util.config(__file__)
    logger = logging.getLogger(__name__)

    args = parse_arguments()
    config = dict(io_utils.json_load(args.config_json))

    try:
        Trainer(config).train()
    except Exception as err:
        logger.error(err, exc_info=True)
        raise err
