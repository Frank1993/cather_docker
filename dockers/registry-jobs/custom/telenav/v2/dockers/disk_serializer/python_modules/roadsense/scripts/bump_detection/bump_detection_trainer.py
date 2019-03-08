import argparse
import logging
from keras.layers import Dense
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam

import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util

from roadsense.scripts.bump_detection.aggregator import HazardAggregator
from roadsense.scripts.general.abstract_trainer import AbstractTrainer
from roadsense.scripts.general.config import ConfigParams as cp


class BumpDetectionTrainer(AbstractTrainer):
    def __init__(self, train_config):
        super().__init__(train_config)
        self.aggregator = HazardAggregator(train_config)

    def add_params_to_config(self, best_threshold, eps, min_samples):
        self.train_config[cp.CONF_THRESHOLD] = best_threshold
        self.train_config[cp.DBSCAN_EPS] = int(eps)
        self.train_config[cp.DBSCAN_MIN_SAMPLES] = int(min_samples)

    def get_model(self, input_dim, nr_classes):
        model = Sequential()
        model.add(Dense(1024, input_dim=input_dim, activation='relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(Dropout(0.3))

        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(Dropout(0.3))

        model.add(Dense(nr_classes, activation='softmax'))
        model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self):
        self.train_model()
        y_pred_proba = self.predict_with_model(self.X_test)
        
        best_conf_thresh = self.get_best_conf_threshold(y_pred_proba)

        self.test_df = self.preprocessor.add_preds_to_df(self.test_df, 
                                                         y_pred_proba, 
                                                         best_conf_thresh, 
                                                         self.class_2_index)
        
        trip_test_df = self.preprocessor.filter_single_trip(self.test_df, "iPhone6")

        eps, min_samples = self.aggregator.get_best_clustering_params(trip_test_df)

        self.add_params_to_config(best_conf_thresh, eps, min_samples)
        self.serialize_model_and_config()


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
        BumpDetectionTrainer(config).train()
    except Exception as err:
        logger.error(err, exc_info=True)
        raise err
