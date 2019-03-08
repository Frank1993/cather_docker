import pandas as pd
from tqdm import tqdm
import numpy as np
import itertools
from functools import partial
from multiprocessing import Pool
import multiprocessing
from sklearn.cluster import DBSCAN

from apollo_python_common.map_geometry.geometry_utils import compute_haversine_distance
from roadsense.scripts.general.config import ConfigParams as cp, Column

class HazardAggregator:
    GPS_DISTANCE_EPS = 30
    TIME_DISTANCE_EPS = 20

    def __init__(self, train_config):
        self.train_config = train_config

    def compute_stats(self, tp_list, fp_list, fn_list, with_print=True):
        tp = len(tp_list)
        fp = len(fp_list)
        fn = len(fn_list)

        prec = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = (2 * prec * recall) / (prec + recall) if prec + recall > 0 else 0

        if with_print:
            print("Prec   = {0:.3f}".format(prec))
            print("Recall = {0:.3f}".format(recall))
            print("F1    = {0:.3f}".format(f1))

        return prec, recall, f1

    def _filter_out_matched_preds(self, pred_centroid_df, matched_pred_ids):
        return pred_centroid_df[~pred_centroid_df[Column.CLUSTER_ID].isin(matched_pred_ids)]

    def _are_clusters_close_enough(self, gps_distance, time_distance):
        return gps_distance < self.GPS_DISTANCE_EPS and time_distance < self.TIME_DISTANCE_EPS

    def _compute_distances_between_clusters(self, gt_row, pred_row):

        gt_lat, gt_lon, gt_datetime = gt_row[Column.CLUSTER_LAT], gt_row[Column.CLUSTER_LON], gt_row[Column.DATETIME]
        pred_lat, pred_lon, pred_datetime = pred_row[Column.CLUSTER_LAT], pred_row[Column.CLUSTER_LON], pred_row[
            Column.DATETIME]

        gps_distance = compute_haversine_distance(pred_lon, pred_lat, gt_lon, gt_lat)
        time_distance = abs((pred_datetime - gt_datetime) / np.timedelta64(1, 's'))
        return gps_distance, time_distance

    def compute_matching(self, gt_centroid_df, pred_centroid_df):
        tp_list, fn_list, matched_pred_ids = [], [], []

        for _, gt_row in gt_centroid_df.iterrows():
            gt_id = gt_row[Column.CLUSTER_ID]
            found = False
            for _, pred_row in self._filter_out_matched_preds(pred_centroid_df, matched_pred_ids).iterrows():
                pred_id = pred_row[Column.CLUSTER_ID]
                gps_distance, time_distance = self._compute_distances_between_clusters(gt_row, pred_row)
                if self._are_clusters_close_enough(gps_distance, time_distance):
                    tp_list.append((gt_id, pred_id))
                    matched_pred_ids.append(pred_id)
                    found = True
                    break

            if not found:
                fn_list.append(gt_id)

        fp_list = self._filter_out_matched_preds(pred_centroid_df, matched_pred_ids)[Column.CLUSTER_ID].values

        return tp_list, fp_list, fn_list

    def compute_cluster_centroids_df(self, clustered_df, lat_col, lon_col):
        data = []
        for cluster_index, cluster_df in clustered_df.groupby(Column.CLUSTER_ID):
            sum_lat = cluster_df[lat_col].astype(np.float32).sum()
            sum_lon = cluster_df[lon_col].astype(np.float32).sum()

            cluster_size = cluster_df.shape[0]

            cluster_lat, cluster_lon = sum_lat / cluster_size, sum_lon / cluster_size
            datetime = cluster_df[Column.DATETIME].values[len(cluster_df) // 2]

            trip_ids = [t.split("_")[-1] for t in cluster_df[Column.TRIP_ID]]
            image_indexes = [int(i) for i in cluster_df[Column.IMAGE_INDEX]]
            member_ids = list(set(zip(trip_ids, image_indexes)))

            data.append((cluster_index, cluster_lat, cluster_lon, datetime, member_ids))

        return pd.DataFrame(data, columns=[Column.CLUSTER_ID, Column.CLUSTER_LAT,
                                           Column.CLUSTER_LON, Column.DATETIME, Column.MEMBER_IDS])

    def perform_clustering(self, df, eps, min_samples):
        features = df.reset_index()[Column.DATETIME].astype(np.int64) // int(1e6)
        features = np.asarray(features).reshape(-1, 1)
        db = DBSCAN(eps, min_samples, algorithm='ball_tree', metric='l1', n_jobs=-1).fit(features)
        df = df.reset_index()
        df[Column.CLUSTER_ID] = pd.Series(db.labels_, index=df.index)
        return df

    def get_centroid_df(self, trip_test_df, eps, min_samples, target_col, lat_col, lon_col,
                        filter_valid_clusters=False):
        print("Aggregating windows...")
        hazard_of_interest = self.train_config[cp.KEPT_HAZARDS][0]  # todo add support for multiple hazards
        hazards_df = trip_test_df[(trip_test_df[target_col] == hazard_of_interest)]
        clustered_df = self.perform_clustering(hazards_df, eps, min_samples)
        if filter_valid_clusters:
            clustered_df = clustered_df[clustered_df[Column.CLUSTER_ID] != -1]
        centroid_df = self.compute_cluster_centroids_df(clustered_df, lat_col, lon_col)
        return centroid_df, clustered_df

    def get_gt_centroid_df(self, trip_test_df):
        return self.get_centroid_df(trip_test_df, 2000, 1, Column.RAW_HAZARD, Column.HAZARD_LAT, Column.HAZARD_LON)

    def get_pred_centroid_df(self, trip_test_df, eps, min_samples):
        return self.get_centroid_df(trip_test_df, eps, min_samples,
                                    Column.PRED, Column.LAT, Column.LON,
                                    filter_valid_clusters=True)

    def compute_metrics_for_params(self, params, trip_test_df, gt_centroid_df):
        print(params)
        eps, min_samples = params
        pred_centroid_df, _ = self.get_pred_centroid_df(trip_test_df, eps, min_samples)
        tp_list, fp_list, fn_list = self.compute_matching(gt_centroid_df, pred_centroid_df)
        prec, recall, f1 = self.compute_stats(tp_list, fp_list, fn_list, with_print=False);
        print("\tPrec {0:.3f}\n\tRec  {1:.3f}\n\tF1   {2:.3f}".format(prec, recall, f1))
        return prec, recall, f1

    def _select_best_thresholds(self, trip_test_df, eps_list, min_samples_list, metric='f1'):
        metric_2_index = {'prec': 0, 'recall': 1, "f1": 2}
        gt_centroid_df, _ = self.get_gt_centroid_df(trip_test_df)
        params_cross_product = list(itertools.product(eps_list, min_samples_list))

        nr_threads = multiprocessing.cpu_count() // 2
        pool = Pool(nr_threads)
        metrics_list = pool.map(partial(self.compute_metrics_for_params,
                                        trip_test_df=trip_test_df,
                                        gt_centroid_df=gt_centroid_df),
                                params_cross_product
                                )
        pool.close()

        params_2_metrics = zip(params_cross_product, metrics_list)
        best_params_2_metrics = sorted(params_2_metrics,
                                       key=lambda param_2_metric: -param_2_metric[1][metric_2_index[metric]])[0]

        best_eps, best_min_samples = best_params_2_metrics[0]
        best_prec, best_recall, best_f1 = best_params_2_metrics[1]

        print("Precision  = {0:.3f}".format(best_prec))
        print("Recall     = {0:.3f}".format(best_recall))
        print("F1         = {0:.3f}".format(best_f1))
        print(f"C_Eps  = {best_eps}")
        print(f"M_Clus = {best_min_samples}")

        return best_eps, best_min_samples

    def get_best_clustering_params(self, trip_test_df):

        print("Computing clustering best params...")
        eps_list = np.arange(500, 1800, 50)
        min_samples_list = np.arange(5, 30, 2)

        eps, min_samples = self._select_best_thresholds(trip_test_df, eps_list, min_samples_list, metric='f1')

        return eps, min_samples
