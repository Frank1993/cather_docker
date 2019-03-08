import os
import logging
import argparse
from tqdm import tqdm
from keras.models import model_from_json

from roadsense.scripts.aggregator import HazardAggregator
from roadsense.scripts.config import ConfigParams as cp, Column
from roadsense.scripts.dataset_preprocessor import DatasetPreprocessor
from roadsense.scripts.hazard_detector import HazardDetector
from roadsense.scripts.meta_aggregator import MetaHazardAggregator

import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util

tqdm.pandas()


class HazardPredictor:
    def __init__(self, bundle_path):
        self.train_config = self.__read_config(bundle_path)
        self.model = self.__load_model_from_json_and_weights(bundle_path)
        self.preprocessor = DatasetPreprocessor(self.train_config)
        self.detector = HazardDetector(self.train_config)
        self.aggregator = HazardAggregator(self.train_config)

    def __read_config(self, bundle_path):
        train_config = io_utils.json_load(os.path.join(bundle_path, "train_config.json"))
        index_2_class_int = {int(k): v for k, v in
                             train_config.index_2_class.items()}  # make keys int. json load problem
        train_config.index_2_class = index_2_class_int
        return train_config

    def __load_model_from_json_and_weights(self, bundle_path):
        json_path = os.path.join(bundle_path, "model_structure.json")
        weights_path = os.path.join(bundle_path, "model_weights.h5")
        with open(json_path, 'r') as json_file:
            loaded_model_json = json_file.read()

        model = model_from_json(loaded_model_json)
        model.load_weights(weights_path)

        return model

    def read_dataset(self, predict_drive_folder):
        X, y, pred_df = self.preprocessor.process_folder(predict_drive_folder)
        y = self.preprocessor.keep_hazards_of_interest(y)
        return X, y, pred_df

    def predict(self, predict_drive_folder, with_gt=True, single_trip_filter=""):
        X, y, pred_df = self.read_dataset(predict_drive_folder)
        y_pred_proba = self.detector.predict_with_model(self.model, X)

        pred_df = self.preprocessor.add_preds_to_df(pred_df.drop([Column.ALL_FEATURES], axis=1),
                                                    y_pred_proba,
                                                    self.train_config[cp.CONF_THRESHOLD])

        if single_trip_filter != "":
            pred_df = self.preprocessor.filter_single_trip(pred_df, single_trip_filter)

        pred_centroid_df, pred_clustered_df = self.aggregator.get_pred_centroid_df(pred_df,
                                                                                   self.train_config[cp.DBSCAN_EPS],
                                                                                   self.train_config[
                                                                                       cp.DBSCAN_MIN_SAMPLES])

        gt_centroid_df, gt_clustered_df = self.aggregator.get_gt_centroid_df(pred_df) if with_gt else (None, None)

        return pred_centroid_df, pred_clustered_df, gt_centroid_df, gt_clustered_df


def save_to_disk(df, output_path):
    print(f"Saving to disk at {output_path}")
    io_utils.create_folder(os.path.dirname(output_path))
    df.to_csv(os.path.join(output_path), index=False)


def save_clusters_to_disk(pred_centroid_df, pred_clustered_df, config):
    save_to_disk(pred_centroid_df, os.path.join(config[cp.PREDICT_OUTPUT_FOLDER], "pred_centroid_df.csv"))
    save_to_disk(pred_clustered_df, os.path.join(config[cp.PREDICT_OUTPUT_FOLDER], "pred_clustered_df.csv"))


def save_meta_clusters_to_disk(meta_pred_df, meta_pred_centroid_df, config):
    save_to_disk(meta_pred_df, os.path.join(config[cp.PREDICT_OUTPUT_FOLDER], "meta_pred_df.csv"))
    save_to_disk(meta_pred_centroid_df, os.path.join(config[cp.PREDICT_OUTPUT_FOLDER], "meta_pred_centroid_df.csv"))


def make_predictions(config):
    predictor = HazardPredictor(config[cp.BUNDLE_PATH])
    pred_centroid_df, pred_clustered_df, gt_centroid_df, gt_clustered_df = predictor.predict(
        config[cp.PREDICT_INPUT_FOLDER],
        with_gt=False,
        single_trip_filter=
        config[cp.SINGLE_TRIP_FILTER]
        )
    save_clusters_to_disk(pred_centroid_df, pred_clustered_df, config)

    if config[cp.WITH_META_AGGREGATION]:
        meta_aggregator = MetaHazardAggregator()
        meta_pred_df, meta_pred_centroid_df = meta_aggregator.get_pred_meta_clusters(pred_centroid_df,
                                                                                     with_valid_filtering=True,
                                                                                     with_min_trips_filtering=
                                                                                     config[cp.WITH_META_MIN_TRIPS_FILTERING])
        save_meta_clusters_to_disk(meta_pred_df, meta_pred_centroid_df, config)


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
        make_predictions(config)
    except Exception as err:
        logger.error(err, exc_info=True)
        raise err
