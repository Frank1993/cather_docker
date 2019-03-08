import os
from keras.models import model_from_json

import apollo_python_common.io_utils as io_utils

from roadsense.scripts.general.dataset_preprocessor import DatasetPreprocessor


class AbstractPredictor:
    def __init__(self, bundle_path):
        self.train_config = self._read_config(bundle_path)
        self.model = self._load_model_from_json_and_weights(bundle_path)
        self.preprocessor = DatasetPreprocessor(self.train_config)

    def _read_config(self, bundle_path):
        train_config = io_utils.json_load(os.path.join(bundle_path, "train_config.json"))
        index_2_class_int = {int(k): v for k, v in
                             train_config.index_2_class.items()}  # make keys int. json load problem
        train_config.index_2_class = index_2_class_int
        return train_config

    def _load_model_from_json_and_weights(self, bundle_path):
        json_path = os.path.join(bundle_path, "model_structure.json")
        weights_path = os.path.join(bundle_path, "model_weights.h5")
        with open(json_path, 'r') as json_file:
            loaded_model_json = json_file.read()

        model = model_from_json(loaded_model_json)
        model.load_weights(weights_path)

        return model

    def read_dataset(self, predict_drive_folders):
        X, y, pred_df = self.preprocessor.process_folders(predict_drive_folders)
        y = self.preprocessor.keep_hazards_of_interest(y)
        return X, y, pred_df
