import argparse
import logging
import os
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm

tqdm.pandas()

import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
from apollo_python_common.map_geometry.geometry_utils import compute_haversine_distance

from roadsense.scripts.general.config import ConfigParams as cp, HazardType as HazardType,Column
from roadsense.scripts.general.abstract_predictor import AbstractPredictor

class RoadQualityPredictor(AbstractPredictor):

    hazard_2_score = {
            HazardType.UNPAVED_ROAD:5,
            HazardType.BUMPY_ROAD:3,
            HazardType.CLEAR:1,
        }
    
    def __init__(self, bundle_path):
        super().__init__(bundle_path)

    def _compute_hazard_score(self, hazard):
        return self.hazard_2_score[hazard]
    
    def _compute_test_acc(self, pred_df):
        same_class_df = pred_df[Column.IS_SAME_CLASS].value_counts()
        tp = same_class_df[True]
        fp = same_class_df[False]
        acc = tp / (tp + fp)
        return acc
    
    def _compute_ws_distance(self,r):
        return compute_haversine_distance(r[Column.SECTION_FROM_NODE_ID_LON],r[Column.SECTION_FROM_NODE_ID_LAT],
                                          r[Column.SECTION_TO_NODE_ID_LON],r[Column.SECTION_TO_NODE_ID_LAT])

    def _get_single_neighbour_ids(self,df, col):
        counts_series = df[col].value_counts()
        counts_series = counts_series[counts_series == 1]
        ids = counts_series.index.values

        return set(ids)

    def _update_pred_class_according_to_neighbours(self, pred_class,from_pred_class,to_pred_class):
        if from_pred_class is None or to_pred_class is None:
            return pred_class

        if from_pred_class == to_pred_class != pred_class:
            return from_pred_class

        return pred_class

    def _uniformize_neighbour_predictions(self, grouped_df):
        neighbour_df = grouped_df.reset_index()

        from_single_neighbour_ids = self._get_single_neighbour_ids(neighbour_df, Column.SECTION_FROM_NODE_ID)
        to_single_neighbour_ids = self._get_single_neighbour_ids(neighbour_df, Column.SECTION_TO_NODE_ID)

        single_neighbour_node_ids = list(from_single_neighbour_ids.intersection(to_single_neighbour_ids))

        from_node_id_df = neighbour_df[[Column.SECTION_FROM_NODE_ID,Column.PRED_ROAD_CLASS]]\
                                .drop_duplicates()\
                                .rename(columns={Column.SECTION_FROM_NODE_ID:Column.FROM_NEIGHBOUR_ID,
                                                 Column.PRED_ROAD_CLASS:Column.FROM_NEIGHBOUR_PRED_CLASS})

        to_node_id_df = neighbour_df[[Column.SECTION_TO_NODE_ID,Column.PRED_ROAD_CLASS]]\
                                .drop_duplicates()\
                                .rename(columns={Column.SECTION_TO_NODE_ID:Column.TO_NEIGHBOUR_ID,
                                                 Column.PRED_ROAD_CLASS:Column.TO_NEIGHBOUR_PRED_CLASS})

        one_neighbour_df = neighbour_df[neighbour_df[Column.SECTION_FROM_NODE_ID].isin(single_neighbour_node_ids) & 
                                        neighbour_df[Column.SECTION_TO_NODE_ID].isin(single_neighbour_node_ids)]

        joined_df = pd.merge(grouped_df.reset_index(),from_node_id_df, how='left',
                             left_on=[Column.SECTION_TO_NODE_ID], 
                             right_on = [Column.FROM_NEIGHBOUR_ID])

        joined_df = pd.merge(joined_df,to_node_id_df, how='left',
                             left_on=[Column.SECTION_FROM_NODE_ID], 
                             right_on = [Column.TO_NEIGHBOUR_ID])

        joined_df = joined_df.drop([Column.FROM_NEIGHBOUR_ID,Column.TO_NEIGHBOUR_ID],axis=1)

        joined_df[Column.PRED_ROAD_CLASS] = joined_df.apply(lambda r: self._update_pred_class_according_to_neighbours(
                                                                        r[Column.PRED_ROAD_CLASS],
                                                                        r[Column.FROM_NEIGHBOUR_PRED_CLASS],
                                                                        r[Column.TO_NEIGHBOUR_PRED_CLASS]),axis=1).astype(int)

        joined_df = joined_df.set_index([Column.WAY_ID,Column.SECTION_FROM_NODE_ID,Column.SECTION_TO_NODE_ID])
        
        return joined_df

    def _add_pred_data(self, pred_df,y_pred_proba):        
        pred_df[Column.PRED_PROBA] = pd.Series([y for y in y_pred_proba], index=pred_df.index)
        pred_df[Column.PRED] = pred_df[Column.PRED_PROBA].apply(lambda p : self.train_config.index_2_class[np.argmax(p)])
        pred_df[Column.PRED_SCORE] = pred_df[Column.PRED].apply(self._compute_hazard_score)
        return pred_df
        
    def _add_gt_data(self, pred_df):
        pred_df.loc[~pred_df[Column.HAZARD].isin(self.train_config[cp.KEPT_HAZARDS]),Column.HAZARD] = HazardType.CLEAR
        pred_df[Column.GT_SCORE] = pred_df[Column.HAZARD].apply(self._compute_hazard_score)
        pred_df[Column.PRED_DIFF] = abs(pred_df[Column.GT_SCORE] - pred_df[Column.PRED_SCORE])
        pred_df[Column.IS_SAME_CLASS] = pred_df[Column.HAZARD] == pred_df[Column.PRED]
        return pred_df
    
    def _filter_data_without_matching(self,pred_df):
        return pred_df[pred_df[Column.SECTION_FROM_NODE_ID] != -1].copy()
    
    def _compute_pred_class_for_ws(self, pred_list):
        counter = Counter(pred_list)
        nr_total = len(pred_list)
        nr_br = counter[HazardType.BUMPY_ROAD]
        nr_unpaved = counter[HazardType.UNPAVED_ROAD]

        if nr_br > nr_unpaved:
            most_common_hazard = HazardType.BUMPY_ROAD
            most_common_hazard_count = nr_br
        else: 
            most_common_hazard = HazardType.UNPAVED_ROAD
            most_common_hazard_count = nr_unpaved

        perc = most_common_hazard_count / nr_total

        perc_threshold = 0.3

        if perc > perc_threshold:
            return self.hazard_2_score[most_common_hazard]

        return self.hazard_2_score[HazardType.CLEAR]

    def _compute_hazard_class(self,score,count):
        return int(score / count)

    
    def _remove_short_hazard_preds(self,r, way_length_thresh = 20):
        return r[Column.PRED_ROAD_CLASS] if r[Column.WAY_LENGTH] >= way_length_thresh \
                                         else self.hazard_2_score[HazardType.CLEAR]

    def _compute_grouped_per_ws_df(self,pred_df):
        

        grouped_df = pred_df.groupby([Column.WAY_ID,Column.SECTION_FROM_NODE_ID,Column.SECTION_TO_NODE_ID])\
                            .agg({Column.PRED_SCORE:"sum",
                                  Column.GT_SCORE:"sum",
                                  Column.WAY_ID:"count",
                                  Column.SECTION_FROM_NODE_ID_LAT:"mean",
                                  Column.SECTION_FROM_NODE_ID_LON:"mean",
                                  Column.SECTION_TO_NODE_ID_LAT:"mean",
                                  Column.SECTION_TO_NODE_ID_LON:"mean"

                                 })

        grouped_df[Column.WAY_LENGTH] = grouped_df.progress_apply(self._compute_ws_distance,axis=1)
        grouped_df[Column.WAY_LENGTH_BUCKET] = grouped_df[Column.WAY_LENGTH] // 10

        grouped_df[Column.PRED_LIST] = pred_df.groupby([Column.WAY_ID,
                                                        Column.SECTION_FROM_NODE_ID,
                                                        Column.SECTION_TO_NODE_ID])[Column.PRED].apply(list)

        grouped_df[Column.TRIP_LIST] = pred_df.groupby([Column.WAY_ID,
                                                        Column.SECTION_FROM_NODE_ID,
                                                        Column.SECTION_TO_NODE_ID])[Column.TRIP_ID].apply(list)
        
        grouped_df[Column.NR_UNIQUE_TRIPS] = grouped_df[Column.TRIP_LIST].apply(lambda l:len(set(l)))

        grouped_df = grouped_df.rename(columns={Column.WAY_ID:Column.COUNT})
        grouped_df[Column.GT_ROAD_CLASS] = grouped_df.progress_apply(lambda r: 
                                                                     self._compute_hazard_class(r[Column.GT_SCORE],
                                                                                                r[Column.COUNT]),axis=1)

        grouped_df[Column.PRED_ROAD_CLASS] = grouped_df[Column.PRED_LIST].apply(self._compute_pred_class_for_ws)
        grouped_df[Column.PRED_ROAD_CLASS] = grouped_df.apply(self._remove_short_hazard_preds, axis=1)
        
        grouped_df = self._uniformize_neighbour_predictions(grouped_df)
        
        grouped_df[Column.ROAD_CLASS_DIFF] = (grouped_df[Column.GT_ROAD_CLASS] - grouped_df[Column.PRED_ROAD_CLASS]).abs()
        grouped_df[Column.IS_SAME_ROAD_CLASS] = grouped_df[Column.GT_ROAD_CLASS] == grouped_df[Column.PRED_ROAD_CLASS]

        return grouped_df
        
    def _compute_way_metrics(self, grouped_df):    
        same_class_df = grouped_df[Column.IS_SAME_ROAD_CLASS].value_counts()
        way_tp = same_class_df[True]
        way_fp = same_class_df[False]
        way_acc = way_tp / (way_tp + way_fp)

        way_class_diff_mean = grouped_df[Column.ROAD_CLASS_DIFF].mean()
        return way_acc, way_class_diff_mean

    def predict(self, predict_drive_folders, with_evaluate = False):
        X, _, pred_df = self.read_dataset(predict_drive_folders)
        y_pred_proba = self.model.predict(X, batch_size=self.train_config[cp.BATCH_SIZE], verbose=1)
        
        pred_df = self._add_pred_data(pred_df,y_pred_proba)
        pred_df = self._add_gt_data(pred_df)
        
        pred_df = self._filter_data_without_matching(pred_df)        
        grouped_df = self._compute_grouped_per_ws_df(pred_df)
        
        if with_evaluate:
            print("Window Metrics")
            print("Accuracy = {}".format(self._compute_test_acc(pred_df)))
            print(classification_report(pred_df[Column.HAZARD], pred_df[Column.PRED], digits=3))
        
            print("Way Section Metrics")
            acc, way_class_diff_mean = self._compute_way_metrics(grouped_df)
            print(f"Accuracy = {acc}")
            print(f"Mean absolute error for class prediction = {way_class_diff_mean}")
           
        return pred_df, grouped_df
    
    
def save_data(pred_df,grouped_df,output_path):
    print("Writing to disk...")
    
    io_utils.create_folder(output_path)
    pred_df.to_csv(os.path.join(output_path,"window_pred_df.csv"))
    grouped_df.drop([Column.PRED_LIST,Column.TRIP_LIST],axis=1).to_csv(os.path.join(output_path,"ws_pred_df.csv"))

def make_predictions(config):
    predictor = RoadQualityPredictor(config[cp.BUNDLE_PATH])
    pred_df, grouped_df = predictor.predict(config[cp.PREDICT_INPUT_FOLDERS],
                                            with_evaluate = config[cp.WITH_EVALUATE])
    
    save_data(pred_df, grouped_df, config[cp.PREDICT_OUTPUT_FOLDER])


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
