import argparse
import math
from collections import defaultdict

import apollo_python_common.io_utils as io_utils
import numpy as np
import pandas as pd
from apollo_python_common.map_geometry.geometry_utils import normalized_angle_difference
from sign_clustering import cluster_plotter as cluster_plotter
from sign_clustering import clustering
from sign_clustering.constants import *
from tqdm import tqdm


class ClusterOutcomeReason:
    VALID = "valid"
    NOT_FORMED = "cluster_not_formed"
    FN_ROI = "fn_roi"


class ClusterStatusReason:
    DEFAULT_CLUSTERING = "default_clustering"
    LOGICAL_CLUSTER_DETECTED = "logical_cluster_detected"


class ClusterEvaluator:

    def __init__(self, evaluation_config):
        self.clustering_config = io_utils.config_load(evaluation_config[CLUSTERING_CONFIG_FILE])
        self.evaluation_config = evaluation_config

    def keep_only_classes_of_interest(self, df, type_col):
        if SELECTED_CLASSES_FILE in self.evaluation_config:
            selected_classes = set(
                io_utils.json_load(self.evaluation_config[SELECTED_CLASSES_FILE])["selected_classes"])
            df = df[df[type_col].isin(selected_classes)]
        return df.copy()

    def read_pred_data(self):
        pred_df = clustering.read_detections_input(self.evaluation_config[PRED_INPUT_FILE],
                                                   self.clustering_config)
        pred_df = self.keep_only_classes_of_interest(pred_df, TYPE_NAME)

        return pred_df

    def add_logical_cluster_id_col(self, gt_df):
        if GT_LOGICAL_CLUSTER_ID not in gt_df.columns:
            gt_df[GT_LOGICAL_CLUSTER_ID] = gt_df[GT_CLUSTER_ID]
        return gt_df

    def read_gt_data(self):
        gt_df = pd.read_csv(self.evaluation_config[GT_FILE])
        gt_df = self.keep_only_classes_of_interest(gt_df, GT_TYPE_NAME)
        gt_df = self.add_logical_cluster_id_col(gt_df)

        return gt_df

    def add_roi_detection_status(self, data_df):
        data_df[ROI_STATUS] = TP
        data_df.loc[data_df[GT_CLUSTER_ID].isnull(), ROI_STATUS] = FP
        data_df.loc[data_df[ROI_X].isnull(), ROI_STATUS] = FN

        return data_df

    def construct_url(self, r):
        trip_id = r[GT_TRIP_ID] if math.isnan(r[TRIP_ID]) else r[TRIP_ID]
        image_index = r[GT_IMAGE_INDEX] if math.isnan(r[IMAGE_INDEX]) else r[IMAGE_INDEX]

        return f"http://openstreetcam.org/details/{int(trip_id)}/{int(image_index)}"

    def merge_data(self, gt_df, rois_df):
        data_df = pd.merge(gt_df, rois_df, how='outer', on=[ROI_ID])
        data_df[URL] = data_df.apply(self.construct_url, axis=1)
        data_df = self.add_roi_detection_status(data_df)

        return data_df

    def filter_small_gt_clusters(self, tp_fn_rois_df):
        return tp_fn_rois_df.groupby([GT_CLUSTER_ID]) \
                            .filter(lambda df: len(df) >= self.clustering_config.min_roi_samples_threshold) \
                            .copy()

    def filter_data(self, data_df):
        tp_rois_df, fp_rois_df, fn_rois_df = self.split_rois_df(data_df)
        tp_fn_rois_df = pd.concat([tp_rois_df, fn_rois_df])

        tp_fn_rois_df = clustering.filter_detections_by_distance(tp_fn_rois_df,
                                                                 self.clustering_config.roi_distance_threshold)
        tp_fn_rois_df = clustering.filter_detections_by_gps_acc(tp_fn_rois_df,
                                                                self.clustering_config.gps_accuracy_threshold)

        tp_fn_rois_df = self.filter_small_gt_clusters(tp_fn_rois_df)

        return pd.concat([tp_fn_rois_df, fp_rois_df])

    def split_rois_df(self, rois_df):
        tp_rois_df = rois_df[rois_df[ROI_STATUS] == TP].copy()
        fp_rois_df = rois_df[rois_df[ROI_STATUS] == FP].copy()
        fn_rois_df = rois_df[rois_df[ROI_STATUS] == FN].copy()

        return tp_rois_df, fp_rois_df, fn_rois_df

    def add_cluster_data(self, data_df):

        tp_rois_df, fp_rois_df, fn_rois_df = self.split_rois_df(data_df)
        tp_fp_rois_df = pd.concat([tp_rois_df, fp_rois_df])

        _, clustered_df = clustering.get_clusters(tp_fp_rois_df,
                                                  self.clustering_config,
                                                  self.evaluation_config[NR_THREADS])

        clustered_df[CLUSTER_OUTCOME] = clustered_df[PRED_CLUSTER_ID].apply(lambda c_id: \
                                                                                ClusterOutcomeReason.NOT_FORMED if c_id == INVALID_CLUSTER_ID else ClusterOutcomeReason.VALID)

        fn_rois_df[PRED_CLUSTER_ID] = INVALID_CLUSTER_ID
        fn_rois_df[CLUSTER_OUTCOME] = ClusterOutcomeReason.FN_ROI

        clustered_df = pd.concat([clustered_df, fn_rois_df])

        return clustered_df

    def get_gt_coords_col_names(self, gt_roi_ids_df):
        column_names = list(gt_roi_ids_df.columns.values)
        if GT_LON in column_names and GT_LAT in column_names:
            lat_col, lon_col = GT_LAT, GT_LON
        else:
            lat_col, lon_col = LAT, LON

        return lat_col, lon_col

    def compute_gt_cluster_coords(self, df):
        lat_col, lon_col = self.get_gt_coords_col_names(df)
        weighted_info = clustering.get_cluster_weighted_info(df, [lat_col, lon_col])
        return weighted_info[lat_col], weighted_info[lon_col]

    def get_gt_clusters_df(self, data_df):
        gt_clusters_grouped = data_df.groupby(GT_CLUSTER_ID)

        gt_clusters_df = pd.DataFrame()
        gt_clusters_df[MEMBER_ROIS] = gt_clusters_grouped[ROI_ID].apply(list)
        gt_clusters_df[GT_TYPE_NAME] = gt_clusters_grouped[GT_TYPE_NAME].apply(lambda g: list(set(g))[0])
        gt_clusters_df[CLUSTER_COORDS] = gt_clusters_grouped.apply(self.compute_gt_cluster_coords)
        gt_clusters_df[GT_LAT] = gt_clusters_df[CLUSTER_COORDS].apply(lambda c: c[0])
        gt_clusters_df[GT_LON] = gt_clusters_df[CLUSTER_COORDS].apply(lambda c: c[1])
        gt_clusters_df[GT_HEADING] = gt_clusters_grouped.apply(lambda df: clustering.get_cluster_heading(df, GT_HEADING))
        gt_clusters_df = gt_clusters_df.drop([CLUSTER_COORDS], axis=1)

        gt_clusters_df = gt_clusters_df.reset_index()
        gt_clusters_df[GT_CLUSTER_ID] = gt_clusters_df[GT_CLUSTER_ID].apply(int)

        return gt_clusters_df

    def get_pred_clusters_df(self, data_df):
        pred_clusters_grouped = data_df.groupby(PRED_CLUSTER_ID)
        valid_pred_clusters_grouped = data_df[data_df[PRED_CLUSTER_ID] != INVALID_CLUSTER_ID].groupby(PRED_CLUSTER_ID)

        all_pred_clusters_df = pd.DataFrame()
        all_pred_clusters_df[MEMBER_ROIS] = pred_clusters_grouped[ROI_ID].apply(list)

        pred_clusters_df = all_pred_clusters_df[all_pred_clusters_df.index != INVALID_CLUSTER_ID].copy()
        pred_clusters_df[TYPE_NAME] = valid_pred_clusters_grouped[TYPE_NAME].apply(lambda g: list(set(g))[0])

        # pred_clusters_df[CLUSTER_COORDS] = valid_pred_clusters_grouped.apply(
        #     lambda df: clustering.get_cluster_location(df, LAT, LON))
        # pred_clusters_df[LAT] = pred_clusters_df[CLUSTER_COORDS].apply(lambda c: c[0])
        # pred_clusters_df[LON] = pred_clusters_df[CLUSTER_COORDS].apply(lambda c: c[1])

        pred_clusters_df[LAT] = valid_pred_clusters_grouped.apply(lambda df: clustering.get_cluster_weighted_info(df, [LAT])[LAT])
        pred_clusters_df[LON] = valid_pred_clusters_grouped.apply(lambda df: clustering.get_cluster_weighted_info(df, [LON])[LON])

        pred_clusters_df[HEADING] = valid_pred_clusters_grouped.apply(
            lambda df: clustering.get_cluster_heading(df, HEADING))

        invalid_cluster_df = all_pred_clusters_df[all_pred_clusters_df.index == INVALID_CLUSTER_ID].copy()

        pred_clusters_df = pred_clusters_df.reset_index()
        invalid_cluster_df = invalid_cluster_df.reset_index()

        return pred_clusters_df, invalid_cluster_df

    def get_match_pairs(self, pred_clusters_df, gt_clusters_df):
        matched_gt_2_pred_dict = defaultdict(lambda: np.nan)

        for _, pred_row in tqdm(pred_clusters_df.iterrows()):
            pred_rois = pred_row[MEMBER_ROIS]
            matched_gt_id = None
            max_matching_ids = 0

            not_matched_clusters_df = gt_clusters_df[~gt_clusters_df[GT_CLUSTER_ID].isin(matched_gt_2_pred_dict.keys())]

            for _, gt_row in not_matched_clusters_df.iterrows():
                nr_matching_ids = len(set(pred_rois) & set(gt_row[MEMBER_ROIS]))

                heading_diff = normalized_angle_difference(pred_row[HEADING], gt_row[GT_HEADING])

                if nr_matching_ids > max_matching_ids and heading_diff <= CLUSTER_ANGLE_DIFF_THRESHOLD:
                    max_matching_ids = nr_matching_ids
                    matched_gt_id = gt_row[GT_CLUSTER_ID]

            if matched_gt_id is not None:
                matched_gt_2_pred_dict[matched_gt_id] = pred_row[PRED_CLUSTER_ID]

        matched_pred_2_gt_dict = defaultdict(lambda: np.nan)
        for k, v in matched_gt_2_pred_dict.items():
            matched_pred_2_gt_dict[v] = k

        return matched_gt_2_pred_dict, matched_pred_2_gt_dict

    def add_match_data(self, data_df, pred_clusters_df, gt_clusters_df):
        matched_gt_2_pred_dict, matched_pred_2_gt_dict = self.get_match_pairs(pred_clusters_df, gt_clusters_df)

        pred_clusters_df[MATCHED_GT_CLUSTER_ID] = pred_clusters_df[PRED_CLUSTER_ID].apply(
            lambda k: matched_pred_2_gt_dict[k])
        gt_clusters_df[MATCHED_PRED_CLUSTER_ID] = gt_clusters_df[GT_CLUSTER_ID].apply(
            lambda k: matched_gt_2_pred_dict[k])

        pred_clusters_df[CLUSTER_STATUS] = pred_clusters_df[MATCHED_GT_CLUSTER_ID].apply(
            lambda gt_id: FP if math.isnan(gt_id) else TP)
        gt_clusters_df[CLUSTER_STATUS] = gt_clusters_df[MATCHED_PRED_CLUSTER_ID].apply(
            lambda pred_id: FN if math.isnan(pred_id) else TP)

        pred_clusters_df[CLUSTER_STATUS_REASON] = ClusterStatusReason.DEFAULT_CLUSTERING
        gt_clusters_df[CLUSTER_STATUS_REASON] = ClusterStatusReason.DEFAULT_CLUSTERING

        gt_clusters_df = self.logical_clusters_status_update(data_df, gt_clusters_df)

        return pred_clusters_df, gt_clusters_df

    def compute_roi_stats(self, data_df):
        vc = data_df[ROI_STATUS].value_counts()
        tp, fp, fn = vc[TP] if TP in vc else 0, vc[FP] if FP in vc else 0, vc[FN] if FN in vc else 0

        print("\n------------ROI Stats------------\n")
        print(f"TP Rois: {tp}")
        print(f"FP Rois: {fp}")
        print(f"FN Rois: {fn} \n")

        precision = round(tp / (tp + fp), 4)
        recall = round(tp / (tp + fn), 4)
        accuracy = round(tp / (tp + fp + fn), 4)

        print(f"Precision {precision}")
        print(f"Recall {recall}")
        print(f"Accuracy {accuracy}")

    def compute_cluster_stats(self, pred_clusters_df, gt_clusters_df):
        pred_vc = pred_clusters_df[CLUSTER_STATUS].value_counts()
        gt_vc = gt_clusters_df[CLUSTER_STATUS].value_counts()
        tp, fp, = pred_vc[TP] if TP in pred_vc else 0, pred_vc[FP] if FP in pred_vc else 0
        fn = gt_vc[FN] if FN in gt_vc else 0

        print("\n------------Cluster Stats------------\n")
        print(f"TP Clusters: {tp}")
        print(f"FP Clusters: {fp}")
        print(f"FN Clusters: {fn} \n")

        precision = round(tp / (tp + fp), 4)
        recall = round(tp / (tp + fn), 4)
        accuracy = round(tp / (tp + fp + fn), 4)

        print(f"Precision {precision}")
        print(f"Recall {recall}")
        print(f"Accuracy {accuracy}")

    def is_lc_found(self, members, gt_cluster_status_dict):
        return any([gt_cluster_status_dict[m] == TP for m in members])

    def lc_status_update(self, r, lc_status_dict):
        if r[CLUSTER_STATUS] == TP:
            return r[CLUSTER_STATUS]

        lc_found = lc_status_dict[r[GT_CLUSTER_ID]]
        return TP if lc_found else FN

    def update_cluster_status_reason(self, r):
        if r[NEW_CLUSTER_STATUS] != r[CLUSTER_STATUS]:
            return ClusterStatusReason.LOGICAL_CLUSTER_DETECTED

        return r[CLUSTER_STATUS_REASON]

    def adjust_cluster_status_according_to_lc(self, gt_clusters_df, lc_status_dict):
        gt_clusters_df[NEW_CLUSTER_STATUS] = gt_clusters_df.apply(lambda r: self.lc_status_update(r, lc_status_dict),
                                                                  axis=1)
        gt_clusters_df[CLUSTER_STATUS_REASON] = gt_clusters_df.apply(self.update_cluster_status_reason, axis=1)
        gt_clusters_df[CLUSTER_STATUS] = gt_clusters_df[NEW_CLUSTER_STATUS]
        gt_clusters_df = gt_clusters_df.drop([NEW_CLUSTER_STATUS], axis=1)

        return gt_clusters_df

    def logical_clusters_status_update(self, data_df, gt_clusters_df):
        gt_cluster_status_dict = {r[GT_CLUSTER_ID]: r[CLUSTER_STATUS] for _, r in gt_clusters_df.iterrows()}

        logical_clusters_df = pd.DataFrame(data_df.groupby(GT_LOGICAL_CLUSTER_ID)[GT_CLUSTER_ID].apply(set))
        logical_clusters_df = logical_clusters_df.rename(columns={GT_CLUSTER_ID: LOGICAL_CLUSTER_MEMBERS})
        logical_clusters_df[LOGICAL_CLUSTER_STATUS] = logical_clusters_df[LOGICAL_CLUSTER_MEMBERS].apply(
            lambda m: self.is_lc_found(m, gt_cluster_status_dict))

        lc_status_dict = {}

        for _, r in logical_clusters_df.iterrows():
            lc_status = r[LOGICAL_CLUSTER_STATUS]
            for gt_cluster_id in r[LOGICAL_CLUSTER_MEMBERS]:
                lc_status_dict[gt_cluster_id] = lc_status

        gt_clusters_df = self.adjust_cluster_status_according_to_lc(gt_clusters_df, lc_status_dict)

        return gt_clusters_df

    def visualize_clusters(self, data_df, gt_clusters_df, pred_clusters_df, invalid_cluster_pred_df):
        if not self.evaluation_config[WITH_VISUALIZATIONS]:
            return

        cluster_plotter.draw_rois(data_df, status=None)
        cluster_plotter.draw_pred_rois_clustered(data_df)
        cluster_plotter.draw_gt_rois_clustered(data_df, status=None)
        cluster_plotter.draw_clusters(gt_clusters_df, pred_clusters_df, invalid_cluster_pred_df, data_df,
                                      status_list=None)

    def compute_stats(self, data_df, pred_clusters_df, gt_clusters_df):
        self.compute_roi_stats(data_df)
        self.compute_cluster_stats(pred_clusters_df, gt_clusters_df)

    def run_evaluate(self, ):
        rois_df = self.read_pred_data()
        gt_df = self.read_gt_data()

        data_df = self.merge_data(gt_df, rois_df)
        data_df = self.filter_data(data_df)
        data_df = self.add_cluster_data(data_df)

        gt_clusters_df = self.get_gt_clusters_df(data_df)
        pred_clusters_df, invalid_cluster_pred_df = self.get_pred_clusters_df(data_df)

        pred_clusters_df, gt_clusters_df = self.add_match_data(data_df, pred_clusters_df, gt_clusters_df)

        self.compute_stats(data_df, pred_clusters_df, gt_clusters_df)

        self.visualize_clusters(data_df, gt_clusters_df, pred_clusters_df, invalid_cluster_pred_df)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--evaluation_config_file", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    eval_config = io_utils.config_load(args.evaluation_config_file)
    ClusterEvaluator(eval_config).run_evaluate()
