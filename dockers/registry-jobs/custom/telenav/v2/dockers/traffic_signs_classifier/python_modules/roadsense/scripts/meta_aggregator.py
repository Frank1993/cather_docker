import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
from sklearn.cluster import DBSCAN

from roadsense.scripts.config import Column
from apollo_python_common.map_geometry.geometry_utils import compute_haversine_distance

tqdm.pandas()


class MetaHazardAggregator:
    META_GPS_DISTANCE_EPS = 20

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

    def __filter_out_matched_preds(self, pred_centroid_df, matched_pred_ids):
        return pred_centroid_df[~pred_centroid_df[Column.META_CLUSTER_ID].isin(matched_pred_ids)]

    def __compute_distances_between_clusters(self, gt_row, pred_row):

        gt_lat, gt_lon = gt_row[Column.META_CLUSTER_LAT], gt_row[Column.META_CLUSTER_LON]
        pred_lat, pred_lon = pred_row[Column.META_CLUSTER_LAT], pred_row[Column.META_CLUSTER_LON]

        gps_distance = compute_haversine_distance(pred_lon, pred_lat, gt_lon, gt_lat)

        return gps_distance

    def compute_meta_matching(self, gt_centroid_df, pred_centroid_df):
        tp_list, fn_list, matched_pred_ids = [], [], []

        for _, gt_row in gt_centroid_df.iterrows():
            gt_id = gt_row[Column.META_CLUSTER_ID]
            found = False
            for _, pred_row in self.__filter_out_matched_preds(pred_centroid_df, matched_pred_ids).iterrows():
                pred_id = pred_row[Column.META_CLUSTER_ID]
                gps_distance = self.__compute_distances_between_clusters(gt_row, pred_row)
                if gps_distance < self.META_GPS_DISTANCE_EPS:
                    tp_list.append((gt_id, pred_id))
                    matched_pred_ids.append(pred_id)
                    found = True
                    break

            if not found:
                fn_list.append(gt_id)

        fp_list = self.__filter_out_matched_preds(pred_centroid_df, matched_pred_ids)[Column.META_CLUSTER_ID].values

        return tp_list, fp_list, fn_list

    def compute_meta_cluster_centroids(self, clustered_df):
        data = []
        for meta_cluster_id, cluster_df in clustered_df.groupby(Column.META_CLUSTER_ID):
            sum_lat = cluster_df[Column.META_CLUSTER_LAT].astype(np.float32).sum()
            sum_lon = cluster_df[Column.META_CLUSTER_LON].astype(np.float32).sum()

            cluster_size = cluster_df.shape[0]

            cluster_lat, cluster_lon = sum_lat / cluster_size, sum_lon / cluster_size

            trip_2_indexes_list_list = cluster_df[Column.MEMBER_IDS]
            trip_2_indexes = list(set([trip_2_index
                                       for trip_2_indexes_list in trip_2_indexes_list_list
                                       for trip_2_index in trip_2_indexes_list]))

            trip_2_indexes = sorted(trip_2_indexes, key=lambda t_2_i: t_2_i[0])

            data.append((meta_cluster_id, cluster_lat, cluster_lon, trip_2_indexes))

        return pd.DataFrame(data, columns=[Column.META_CLUSTER_ID, Column.META_CLUSTER_LAT,
                                           Column.META_CLUSTER_LON, Column.MEMBER_IDS])

    def has_enough_trips(self, member_ids, trip_threshold):
        trip_ids = [trip_id for trip_id, image_index in member_ids]
        return len(set(trip_ids)) >= trip_threshold

    def get_meta_cluster_centroids(self, centroid_df, eps, min_samples):
        meta_df = centroid_df[[Column.CLUSTER_LAT, Column.CLUSTER_LON, Column.MEMBER_IDS]].copy()
        meta_df = meta_df.rename(columns={Column.CLUSTER_LAT: Column.META_CLUSTER_LAT,
                                          Column.CLUSTER_LON: Column.META_CLUSTER_LON})
        centroids = meta_df[[Column.META_CLUSTER_LAT, Column.META_CLUSTER_LON]].values
        db = DBSCAN(eps=eps / 6371., min_samples=min_samples,
                    algorithm='ball_tree', metric='haversine').fit(np.radians(centroids))
        meta_df[Column.META_CLUSTER_ID] = pd.Series(db.labels_, index=meta_df.index)
        return meta_df

    def get_meta_clusters(self, centroid_id, eps, min_samples, with_valid_filtering=False):
        meta_df = self.get_meta_cluster_centroids(centroid_id, eps=eps, min_samples=min_samples)
        if with_valid_filtering:
            meta_df = meta_df[meta_df[Column.META_CLUSTER_ID] != -1]
        meta_centroid_df = self.compute_meta_cluster_centroids(meta_df)
        return meta_df, meta_centroid_df

    def get_gt_meta_clusters(self, centroid_df):
        return self.get_meta_clusters(centroid_df, 0.01, 1)

    def get_pred_meta_clusters(self, centroid_df, with_valid_filtering=False, with_min_trips_filtering=False):
        meta_df, meta_centroid_df = self.get_meta_clusters(centroid_df, 0.015, 2,
                                                           with_valid_filtering=with_valid_filtering)
        meta_centroid_df[Column.HAS_ENOUGH_TRIPS] = meta_centroid_df[Column.MEMBER_IDS] \
            .apply(partial(self.has_enough_trips,
                           trip_threshold=2))

        if with_min_trips_filtering:
            meta_centroid_df = meta_centroid_df[meta_centroid_df[Column.HAS_ENOUGH_TRIPS]]
            meta_df = pd.merge(meta_df, meta_centroid_df[[Column.META_CLUSTER_ID]], on=Column.META_CLUSTER_ID,
                               how='inner')

        return meta_df, meta_centroid_df
