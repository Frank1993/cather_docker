import argparse
import logging
from keras.layers import Dense
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam

from roadsense.scripts.general.abstract_trainer import AbstractTrainer

import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util


class RoadQualityTrainer(AbstractTrainer):
    def __init__(self, train_config):
        super().__init__(train_config)

    def get_model(self, input_dim, nr_classes):
        model = Sequential()
        model.add(Dense(128, input_dim=input_dim, activation='relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(Dropout(0.7))

        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(Dropout(0.7))

        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(Dropout(0.7))

        model.add(Dense(nr_classes, activation='softmax'))
        model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self):
        self.train_model()        
        acc = round(self.model.evaluate(self.X_test,self.y_test_ohe)[1],4)
        
        print(f"Accuracy: {acc}")
        
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
        RoadQualityTrainer(config).train()
    except Exception as err:
        logger.error(err, exc_info=True)
        raise err
    