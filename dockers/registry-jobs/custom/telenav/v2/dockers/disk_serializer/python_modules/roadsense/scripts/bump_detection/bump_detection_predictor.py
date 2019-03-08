import argparse
import logging
import os

import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
from roadsense.scripts.bump_detection.aggregator import HazardAggregator
from roadsense.scripts.bump_detection.meta_aggregator import MetaHazardAggregator
from roadsense.scripts.general.abstract_predictor import AbstractPredictor
from roadsense.scripts.general.config import ConfigParams as cp


class BumpDetectionPredictor(AbstractPredictor):
    def __init__(self, bundle_path):
        super().__init__(bundle_path)
        self.aggregator = HazardAggregator(self.train_config)

    def predict(self, predict_drive_folders, with_gt=True, single_trip_filter=""):
        X, _, pred_df = self.read_dataset(predict_drive_folders)
        y_pred_proba = self.model.predict(X, batch_size=self.train_config[cp.BATCH_SIZE], verbose=1)
        
        pred_df = self.preprocessor.add_preds_to_df(pred_df, 
                                                    y_pred_proba, 
                                                    self.train_config[cp.CONF_THRESHOLD],
                                                    self.train_config.class_2_index)

        if single_trip_filter != "":
            pred_df = self.preprocessor.filter_single_trip(pred_df, single_trip_filter)

        pred_centroid_df, pred_clustered_df = self.aggregator.get_pred_centroid_df(pred_df,
                                                                                   self.train_config[cp.DBSCAN_EPS],
                                                                                   self.train_config[cp.DBSCAN_MIN_SAMPLES])

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
    predictor = BumpDetectionPredictor(config[cp.BUNDLE_PATH])
    pred_centroid_df, pred_clustered_df, gt_centroid_df, gt_clustered_df = predictor.predict(
        config[cp.PREDICT_INPUT_FOLDERS],
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
