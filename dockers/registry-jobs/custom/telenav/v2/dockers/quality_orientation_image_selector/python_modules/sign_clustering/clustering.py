import argparse
import logging
import math
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from functools import partial

tqdm.pandas()

from sklearn.cluster import DBSCAN

import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
import apollo_python_common.proto_api as proto_api
from sign_clustering.constants import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=str, required=True)
    parser.add_argument("-c", "--config_file", type=str, required=True)
    parser.add_argument("-o", "--output_folder", type=str, required=True)
    parser.add_argument("-t", "--threads_nr", type=int, default=4)
    return parser.parse_args()


def explode_rois(data_df, column_name):
    rois_df_cols = list(data_df.columns)
    rois_df_cols.append(ROI)

    rows = []
    for _, row in data_df.iterrows():
        for nn in row[column_name]:
            rows.append(row.tolist() + [nn])

    rois_df = pd.DataFrame(rows, columns=rois_df_cols)
    rois_df = rois_df.drop(column_name, axis=1)

    return rois_df


def filter_detections_by_angle(rois_df, angle_threshold):
    return rois_df[rois_df[ANGLE_OF_ROI] < angle_threshold]


def filter_detections_by_size(rois_df, size_threshold):
    return rois_df[(rois_df[ROI_WIDTH] > size_threshold) & (rois_df[ROI_HEIGHT] > size_threshold)]


def filter_detections_by_distance(rois_df, distance_threshold):
    return rois_df[rois_df[DISTANCE] < distance_threshold]


def filter_detections_by_gps_acc(rois_df, accuracy_threshold):
    return rois_df[rois_df[GPS_ACC] < accuracy_threshold]


def normalized_heading(orig_heading):
    if orig_heading < 0:
        orig_heading += 360
    return orig_heading


def get_weight(row, config):
    image_area = row[IMAGE_WIDTH] * row[IMAGE_HEIGHT]
    roi_area = row[ROI_WIDTH] * row[ROI_HEIGHT]
    roi_area_percentage = min(roi_area / image_area * 100, config.high_weight_area_percentage)

    return roi_area_percentage / config.high_weight_area_percentage * config.cluster_weight_threshold


def clusters_to_dataframe(input_clusters):
    det_df = pd.DataFrame({CLUSTER_PROTO: list(input_clusters)})

    det_df.loc[:, ROIS] = det_df.loc[:, CLUSTER_PROTO].apply(lambda cluster_proto: cluster_proto.roi_ids)
    det_df.loc[:, PREDICTED_CLUSTER_LABEL] = range(len(det_df))
    det_df.loc[:, TYPE] = det_df.loc[:, CLUSTER_PROTO].apply(lambda cluster_proto: cluster_proto.type)
    det_df.loc[:, CONFIDENCE] = det_df.loc[:, CLUSTER_PROTO].apply(lambda cluster_proto: cluster_proto.confidence)
    det_df.loc[:, LATITUDE] = det_df.loc[:, CLUSTER_PROTO].apply(lambda cluster_proto: cluster_proto.location.latitude)
    det_df.loc[:, LONGITUDE] = det_df.loc[:, CLUSTER_PROTO].apply(lambda cluster_proto: cluster_proto.location.longitude)
    det_df.loc[:, FACING] = det_df.loc[:, CLUSTER_PROTO].apply(lambda cluster_proto: cluster_proto.facing)
    det_df.loc[:, WEIGHT] = 1
    det_df = explode_rois(det_df, ROIS)
    det_df = det_df.rename(columns={ROI: ID})

    det_df = det_df.drop([CLUSTER_PROTO], axis=1)
    return det_df


def image_set_to_dataframe(input_image_set, config):
    det_df = pd.DataFrame({IMAGE_PROTO: list(input_image_set.images)})

    det_df.loc[:, ROIS] = det_df.loc[:, IMAGE_PROTO].apply(lambda im_proto: im_proto.rois)
    det_df.loc[:, TRIP_ID] = det_df.loc[:, IMAGE_PROTO].apply(lambda im_proto: im_proto.metadata.trip_id)
    det_df.loc[:, IMAGE_INDEX] = det_df.loc[:, IMAGE_PROTO].apply(lambda im_proto: im_proto.metadata.image_index)
    det_df.loc[:, IMAGE_WIDTH] = det_df.loc[:, IMAGE_PROTO].apply(lambda im_proto: im_proto.sensor_data.img_res.width)
    det_df.loc[:, IMAGE_HEIGHT] = det_df.loc[:, IMAGE_PROTO].apply(lambda im_proto: im_proto.sensor_data.img_res.height)
    det_df.loc[:, GPS_ACC] = det_df.loc[:, IMAGE_PROTO].apply(lambda im_proto: im_proto.sensor_data.gps_accuracy)
    det_df = explode_rois(det_df, column_name=ROIS)
    det_df.loc[:, ID] = det_df.loc[:, ROI].apply(lambda roi: roi.id)
    det_df.loc[:, TYPE] = det_df.loc[:, ROI].apply(lambda roi: roi.type)
    det_df.loc[:, CONFIDENCE] = det_df.loc[:, ROI].apply(lambda roi: roi.detections[0].confidence)
    det_df.loc[:, ROI_X] = det_df.loc[:, ROI].apply(lambda roi: roi.rect.tl.col)
    det_df.loc[:, ROI_Y] = det_df.loc[:, ROI].apply(lambda roi: roi.rect.tl.row)
    det_df.loc[:, ROI_WIDTH] = det_df.loc[:, ROI].apply(lambda roi: roi.rect.br.col - roi.rect.tl.col)
    det_df.loc[:, ROI_HEIGHT] = det_df.loc[:, ROI].apply(lambda roi: roi.rect.br.row - roi.rect.tl.row)
    det_df.loc[:, LATITUDE] = det_df.loc[:, ROI].apply(lambda roi: roi.local.position.latitude)
    det_df.loc[:, LONGITUDE] = det_df.loc[:, ROI].apply(lambda roi: roi.local.position.longitude)
    det_df.loc[:, DISTANCE] = det_df.loc[:, ROI].apply(lambda roi: roi.local.distance / 1000)  # from mm to m
    det_df.loc[:, ANGLE_OF_ROI] = det_df.loc[:, ROI].apply(lambda roi: roi.local.angle_of_roi)
    det_df.loc[:, FACING] = det_df.loc[:, ROI].apply(lambda roi: roi.local.facing)
    det_df.loc[:, WEIGHT] = det_df.apply(partial(get_weight, config=config), axis=1)
    det_df = det_df.drop([IMAGE_PROTO, ROI], axis=1)

    return det_df


def read_detections_input(input_file, config):
    input_image_set = proto_api.read_imageset_file(input_file)
    return image_set_to_dataframe(input_image_set, config)


def get_cartezian_coords_in_batch(geo_coords):
    # from http://earthpy.org/tag/scipy.html
    R = np.float64(6378137)  # radius of earth in meters
    latitude = np.radians(geo_coords[:, 0].astype(np.float64))
    longitude = np.radians(geo_coords[:, 1].astype(np.float64))
    X = R * np.cos(latitude) * np.cos(longitude)
    Y = R * np.cos(latitude) * np.sin(longitude)
    Z = R * np.sin(latitude)
    return np.column_stack((X, Y, Z)).astype(np.float32)


def get_features(df, facing_factor):
    geo_coords = df[[LATITUDE, LONGITUDE]].values
    cartezian_coords = get_cartezian_coords_in_batch(geo_coords)

    type_feature = df[TYPE].values * 100000
    facing_feature = np.clip(df[FACING].values, 0, 360)
    facing_feature = np.column_stack((np.sin(np.radians(facing_feature)),
                                      np.cos(np.radians(facing_feature)))) * facing_factor

    features = np.column_stack((cartezian_coords, type_feature, facing_feature))
    return features


def get_clusters(detections_df, config, threads_number):
    cluster_labels = get_cluster_labels(detections_df, config, threads_number)
    clustered_df = detections_df.copy()
    clustered_df[PREDICTED_CLUSTER_LABEL] = cluster_labels
    cluster_count = len(set(cluster_labels))
    return cluster_count, clustered_df


def get_cluster_labels(detections_df, config, threads_number):
    features = get_features(detections_df, int(config.facing_factor))
    sample_weights = np.asarray(detections_df[WEIGHT])
    db = DBSCAN(eps=int(config.dbscan_epsilon), min_samples=config.cluster_weight_threshold, algorithm='ball_tree',
                metric='euclidean', n_jobs=threads_number).fit(features, sample_weight=sample_weights)
    return db.labels_


def create_cluster_proto(detections_df, config):
    proto_clusters = proto_api.get_new_cluster_proto()

    grouped = detections_df.groupby([PREDICTED_CLUSTER_LABEL])
    for cluster_label, cluster_df in tqdm(grouped):
        if cluster_label == INVALID_CLUSTER_ID:
            continue
        cluster_size = cluster_df.shape[0]
        if cluster_size < int(config.min_roi_samples_threshold):
            continue
        new_cluster = proto_clusters.cluster.add()
        sum_long = 0
        sum_lat = 0
        sum_cos = 0
        sum_sin = 0
        sum_conf = 0
        weight_sum = 0
        for _, row in cluster_df.iterrows():
            det_id = int(row[ID])
            det_type = int(row[TYPE])
            det_confidence = row[CONFIDENCE]
            det_long = row[LONGITUDE]
            det_lat = row[LATITUDE]
            det_weight = row[WEIGHT]
            weight_sum += det_weight
            det_facing = row[FACING]
            new_cluster.roi_ids.append(det_id)
            sum_long += det_long * det_weight
            sum_lat += det_lat * det_weight
            sum_cos += math.cos(math.radians(det_facing)) * det_weight
            sum_sin += math.sin(math.radians(det_facing)) * det_weight
            sum_conf += det_confidence
        new_cluster.location.longitude = sum_long / weight_sum
        new_cluster.location.latitude = sum_lat / weight_sum
        # https://docs.python.org/3/library/math.html?highlight=atan2#math.atan2
        new_cluster.facing = normalized_heading(math.degrees(math.atan2(sum_sin / weight_sum, sum_cos / weight_sum)))
        new_cluster.type = det_type
        new_cluster.confidence = sum_conf / cluster_size
        new_cluster.algorithm = config.algorithm
        new_cluster.algorithm_version = config.algorithm_version
    return proto_clusters


if __name__ == "__main__":
    log_util.config(__file__)
    logger = logging.getLogger(__name__)

    args = get_args()
    input_file = args.input_file
    config_file = args.config_file
    output_folder = args.output_folder
    nr_threads = args.threads_nr

    config = io_utils.config_load(config_file)

    start_time = time.time()
    detections_df = read_detections_input(input_file, config)
    logger.info("Completed reading input rois in {} seconds".format(time.time() - start_time))
    logger.info('Number of signs: {}'.format(detections_df.shape[0]))

    detections_df = filter_detections_by_distance(detections_df, config.roi_distance_threshold)
    detections_df = filter_detections_by_gps_acc(detections_df, config.gps_accuracy_threshold)

    start_time = time.time()
    num_clusters, clustered_df = get_clusters(detections_df, config, nr_threads)
    logger.info("Completed clustering of rois in {} seconds".format(time.time() - start_time))
    logger.info('Number of clusters: {:,}'.format(num_clusters))

    start_time = time.time()
    proto_clusters = create_cluster_proto(clustered_df, config)
    logger.info("Completed writing clusters to proto in {} seconds".format(time.time() - start_time))

    proto_api.serialize_proto_instance(proto_clusters, output_folder, 'out_clusters')
