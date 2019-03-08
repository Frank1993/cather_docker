import argparse
import logging
import numpy as np
import pandas as pd
import time
from math import radians, cos, sin, atan2, degrees
from tqdm import tqdm
from functools import partial

tqdm.pandas()

from sklearn.cluster import DBSCAN
from pysal.cg.kdtree import KDTree
import pysal

import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
import apollo_python_common.proto_api as proto_api
from apollo_python_common.map_geometry.geometry_utils import normalized_angle_difference
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


def get_cluster_heading(cluster_rois_df, heading_col):
    sum_cos = 0
    sum_sin = 0
    weight_sum = 0
    for _, row in cluster_rois_df.iterrows():
        det_weight = row[WEIGHT] if WEIGHT in cluster_rois_df.columns else 1
        det_heading = row[heading_col]
        weight_sum += det_weight
        sum_cos += cos(radians(det_heading)) * det_weight
        sum_sin += sin(radians(det_heading)) * det_weight
    heading = normalized_heading(degrees(atan2(sum_sin / weight_sum, sum_cos / weight_sum)))
    return heading


def get_cluster_weighted_info(cluster_rois_df, columns=[LAT, LON, CONFIDENCE]):
    results = {}
    for column_name in columns:
        partial_result = (cluster_rois_df[column_name] *
                          cluster_rois_df[WEIGHT]).sum() / cluster_rois_df[WEIGHT].sum() \
            if WEIGHT in cluster_rois_df else cluster_rois_df[column_name].mean()
        results[column_name] = partial_result
    return results


def clusters_to_dataframe(input_clusters):
    det_df = pd.DataFrame({CLUSTER_PROTO: list(input_clusters)})

    det_df[ROIS] = det_df[CLUSTER_PROTO].apply(lambda cluster_proto: cluster_proto.roi_ids)
    det_df[PRED_CLUSTER_ID] = range(len(det_df))
    det_df[TYPE] = det_df[CLUSTER_PROTO].apply(lambda cluster_proto: cluster_proto.type)
    det_df[CONFIDENCE] = det_df[CLUSTER_PROTO].apply(lambda cluster_proto: cluster_proto.confidence)
    det_df[LAT] = det_df[CLUSTER_PROTO].apply(lambda cluster_proto: cluster_proto.location.latitude)
    det_df[LON] = det_df[CLUSTER_PROTO].apply(lambda cluster_proto: cluster_proto.location.longitude)
    det_df[HEADING] = det_df[CLUSTER_PROTO].apply(lambda cluster_proto: cluster_proto.facing)
    det_df[WEIGHT] = 1
    det_df = explode_rois(det_df, ROIS)
    det_df = det_df.rename(columns={ROI: ROI_ID})

    det_df = det_df.drop([CLUSTER_PROTO], axis=1)
    return det_df


def image_set_to_dataframe(input_image_set, config):
    det_df = pd.DataFrame({IMAGE_PROTO: list(input_image_set.images)})

    det_df[ROIS] = det_df[IMAGE_PROTO].apply(lambda im_proto: im_proto.rois)
    det_df[TRIP_ID] = det_df[IMAGE_PROTO].apply(lambda im_proto: int(im_proto.metadata.trip_id))
    det_df[IMAGE_INDEX] = det_df[IMAGE_PROTO].apply(lambda im_proto: im_proto.metadata.image_index)
    det_df[IMAGE_WIDTH] = det_df[IMAGE_PROTO].apply(lambda im_proto: im_proto.sensor_data.img_res.width)
    det_df[IMAGE_HEIGHT] = det_df[IMAGE_PROTO].apply(lambda im_proto: im_proto.sensor_data.img_res.height)
    det_df[GPS_ACC] = det_df[IMAGE_PROTO].apply(lambda im_proto: im_proto.sensor_data.gps_accuracy)
    det_df = explode_rois(det_df, column_name=ROIS)
    det_df[ROI_ID] = det_df[ROI].apply(lambda roi: roi.id)
    det_df[TYPE] = det_df[ROI].apply(lambda roi: roi.type)
    det_df[TYPE_NAME] = det_df[TYPE].apply(proto_api.get_roi_type_name)
    det_df[CONFIDENCE] = det_df[ROI].apply(lambda roi: roi.detections[0].confidence)
    det_df[ROI_X] = det_df[ROI].apply(lambda roi: roi.rect.tl.col)
    det_df[ROI_Y] = det_df[ROI].apply(lambda roi: roi.rect.tl.row)
    det_df[ROI_WIDTH] = det_df[ROI].apply(lambda roi: roi.rect.br.col - roi.rect.tl.col)
    det_df[ROI_HEIGHT] = det_df[ROI].apply(lambda roi: roi.rect.br.row - roi.rect.tl.row)
    det_df[LAT] = det_df[ROI].apply(lambda roi: roi.local.position.latitude)
    det_df[LON] = det_df[ROI].apply(lambda roi: roi.local.position.longitude)
    det_df[DISTANCE] = det_df[ROI].apply(lambda roi: roi.local.distance / 1000)  # from mm to m
    det_df[ANGLE_OF_ROI] = det_df[ROI].apply(lambda roi: roi.local.angle_of_roi)
    det_df[HEADING] = det_df[ROI].apply(lambda roi: roi.local.facing) # roi.local.facing actually stores the heading
    det_df[WEIGHT] = det_df.apply(partial(get_weight, config=config), axis=1)
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


def get_features(df, heading_factor):
    geo_coords = df[[LAT, LON]].values
    cartezian_coords = get_cartezian_coords_in_batch(geo_coords)

    type_feature = df[TYPE].values * 100000
    heading_feature = np.clip(df[HEADING].values, 0, 360)
    heading_feature = np.column_stack((np.sin(np.radians(heading_feature)),
                                      np.cos(np.radians(heading_feature)))) * heading_factor

    features = np.column_stack((cartezian_coords, type_feature, heading_feature))
    return features


def can_reduce_cluster(cluster_df, config):
    """
    Method that indicates if a cluster can be reduced to less rois without becoming irelevant
    :param cluster_df: dataframe containing rois of one cluster
    :param config: config containing used thresholds
    :return: True if cluster can be reduced, False otherwise
    """
    max_distance = cluster_df[DISTANCE].max()
    return (len(cluster_df) > config.large_cluster_threshold) & (max_distance > config.large_cluster_distance_threshold)


def reduce_cluster(cluster_df, config):
    """
    Reduces the cluster size with a percentage of it's size up to a limit(either up to a distance limit or a threshold)
    :param cluster_df: dataframe containing rois of one cluster
    :param config: config containing used thresholds
    :return: returns the filtered and reduced cluster dataframe
    """
    sorted_cluster_df = cluster_df.sort_values(DISTANCE).reset_index(drop=True)
    big_distance_roi_indexes = sorted_cluster_df[sorted_cluster_df[DISTANCE] >
                                                 config.large_cluster_distance_threshold].index.values
    reducing_index = max(big_distance_roi_indexes[0],
                         config.large_cluster_threshold,
                         int(len(cluster_df) * (1 - config.large_cluster_reducing_factor)))
    reduced_cluster_df = sorted_cluster_df[:reducing_index]
    return reduced_cluster_df


def recluster_rois(cluster_df, config):
    """
    Does reclustering of a dataframe containing rois and keeps only the valid predictions
    :param cluster_df: dataframe containing roi info and predicted cluster labels
    :param config: config for the needed thresholds
    :return: number of cluster labels and the resulting dataframe
    """
    cluster_labels = get_cluster_labels(cluster_df, config, 1)
    cluster_df[RECLUSTERING_LABEL] = cluster_labels
    cluster_df = cluster_df[cluster_df[RECLUSTERING_LABEL] != INVALID_CLUSTER_ID]
    unique_labels = cluster_df[RECLUSTERING_LABEL].unique()
    return len(unique_labels), cluster_df


def split_large_clusters(clustered_df, config):
    """
    Try to split large clusters into multiple clusters in case they are wrongly joined
    :param clustered_df: dataframe containing roi info and predicted cluster labels
    :param config: config for the needed thresholds
    :return: returns the modified dataframe in which large clusters have been split and trimmed
    """
    clustered_df[PARENT_CLUSTER_ID] = ''
    grouped_pred_clusters = clustered_df.groupby([PRED_CLUSTER_ID])
    cluster_list = []
    max_predicted_id = clustered_df[PRED_CLUSTER_ID].max() + 1

    for pred_cluster_id, rois_df in grouped_pred_clusters:
        if pred_cluster_id == INVALID_CLUSTER_ID:
            continue
        cluster_was_split = False
        while can_reduce_cluster(rois_df, config):
            reduced_cluster_df = reduce_cluster(rois_df, config)
            nr_clusters, reclustered_rois_df = recluster_rois(reduced_cluster_df, config)
            if nr_clusters > 1:
                reclustered_rois_df[PARENT_CLUSTER_ID] = reclustered_rois_df.apply(
                    lambda row: str(row[PRED_CLUSTER_ID])
                    if row[PARENT_CLUSTER_ID] == ''
                    else "{}_{}".format(row[PARENT_CLUSTER_ID], row[RECLUSTERING_LABEL]), axis=1)
                reclustered_rois_df[PRED_CLUSTER_ID] = reclustered_rois_df.apply(
                    lambda row: max_predicted_id + row[RECLUSTERING_LABEL], axis=1)
                reclustered_rois_df.drop([RECLUSTERING_LABEL], axis=1)
                split_clusters_df = split_large_clusters(reclustered_rois_df, config)
                cluster_list.append(split_clusters_df)
                cluster_was_split = True
                max_predicted_id = max_predicted_id + nr_clusters
                break
            reclustered_rois_df = reclustered_rois_df.drop([RECLUSTERING_LABEL], axis=1)
            rois_df = reclustered_rois_df
        if not cluster_was_split:
            cluster_list.append(rois_df)

    split_df = pd.concat(cluster_list, sort=False)
    return split_df


def build_density_tree(cluster_df):
    locations_long = list(map(float, cluster_df[[LON]].values))
    locations_lat = list(map(float, cluster_df[[LAT]].values))
    locations = list(zip(locations_long, locations_lat))

    # build the density tree using every roi location
    density_tree = KDTree(locations, distance_metric='Arc', radius=pysal.cg.RADIUS_EARTH_KM)
    return density_tree


def filter_predictions_in_list(cluster_df, filter_list):
    cluster_df.loc[cluster_df[PRED_CLUSTER_ID].isin(filter_list),
                   PRED_CLUSTER_ID] = INVALID_CLUSTER_ID
    return cluster_df


def filter_by_density(cluster_df, density_radius, density_percentage_threshold):
    """
    Filter by density(small size clusters will be marked invalid if they are in high roi density areas)
    :param cluster_df: dataframe containing roi info and predicted cluster labels
    :param density_radius: radius of a vicinity in which to query the density(in km)
    :param density_percentage_threshold: density percentage threshold that a predicted cluster must pass
    :return: returns the input dataframe modified with density based invalidated predicted labels
    """
    cluster_df.reset_index(drop=True, inplace=True)
    density_tree = build_density_tree(cluster_df)
    filtered_list = []
    grouped_pred_clusters = cluster_df.groupby([PRED_CLUSTER_ID])

    predicted_cluster_headings = {}
    for pred_cluster_id, rois_df in grouped_pred_clusters:
        predicted_cluster_headings[pred_cluster_id] = get_cluster_heading(rois_df, HEADING)

    for pred_cluster_id, rois_df in grouped_pred_clusters:
        if pred_cluster_id == INVALID_CLUSTER_ID:
            continue
        # query the vicinity of each predicted cluster to get the list of its close neighbours
        cluster_info = get_cluster_weighted_info(rois_df, columns=[LAT, LON])
        indices = density_tree.query_ball_point((cluster_info[LON], cluster_info[LAT]), density_radius)
        if len(indices) == 0:
            continue

        # compute a weighted average density based on the number of rois for each cluster in the vicinity
        cluster_heading = predicted_cluster_headings[pred_cluster_id]
        weight_sum = 0
        roi_count = 0
        grouped_clusters_in_vicinity = cluster_df.loc[indices].groupby([PRED_CLUSTER_ID])
        for cluster_id, cluster_rois_df in grouped_clusters_in_vicinity:
            if normalized_angle_difference(predicted_cluster_headings[cluster_id],
                                           cluster_heading) < CLUSTER_ANGLE_DIFF_THRESHOLD:
                cluster_size = len(cluster_rois_df)
                weight_sum += cluster_size * cluster_size
                roi_count += cluster_size
        if roi_count == 0:
            continue
        weighted_avg = weight_sum / roi_count

        # filter the cluster predictions that have a size smaller than a percentage of the weighted average
        if len(rois_df) < density_percentage_threshold * weighted_avg:
            filtered_list.append(pred_cluster_id)
    cluster_df = filter_predictions_in_list(cluster_df, filtered_list)
    return cluster_df


def filter_by_size(cluster_df, min_size):
    if len(cluster_df) < min_size:
        cluster_df[PRED_CLUSTER_ID] = INVALID_CLUSTER_ID
    return cluster_df


def filter_clusters_by_nr_of_samples(clustered_df, min_samples):
    filtered_clusters_df = clustered_df.groupby([PRED_CLUSTER_ID]).apply(
        partial(filter_by_size, min_size=min_samples))
    return filtered_clusters_df


def get_clusters(detections_df, config, threads_number):
    cluster_labels = get_cluster_labels(detections_df, config, threads_number)
    clustered_df = detections_df.copy()
    clustered_df[PRED_CLUSTER_ID] = cluster_labels

    if config.with_large_cluster_splitting:
        clustered_df = split_large_clusters(clustered_df, config)

    if config.with_density_filter:
        clustered_df = filter_by_density(clustered_df, config.density_radius, config.density_percentage_threshold)

    # filter by size(clusters smaller than min_roi_samples_threshold will be marked invalid)
    filtered_clusters_df = filter_clusters_by_nr_of_samples(clustered_df, config.min_roi_samples_threshold)

    cluster_count = len(filtered_clusters_df[PRED_CLUSTER_ID].unique())
    return cluster_count, filtered_clusters_df


def get_cluster_labels(detections_df, config, threads_number):
    features = get_features(detections_df, int(config.heading_factor))
    sample_weights = np.asarray(detections_df[WEIGHT])
    db = DBSCAN(eps=int(config.dbscan_epsilon), min_samples=config.cluster_weight_threshold, algorithm='ball_tree',
                metric='euclidean', n_jobs=threads_number).fit(features, sample_weight=sample_weights)
    return db.labels_


def create_cluster_proto(detections_df, config):
    proto_clusters = proto_api.get_new_cluster_proto()

    grouped = detections_df.groupby([PRED_CLUSTER_ID])
    for cluster_label, cluster_df in tqdm(grouped):
        if cluster_label == INVALID_CLUSTER_ID:
            continue
        cluster_size = cluster_df.shape[0]
        new_cluster = proto_clusters.cluster.add()
        sum_long = 0
        sum_lat = 0
        sum_cos = 0
        sum_sin = 0
        sum_conf = 0
        weight_sum = 0
        for _, row in cluster_df.iterrows():
            det_id = int(row[ROI_ID])
            det_type = int(row[TYPE])
            det_confidence = row[CONFIDENCE]
            det_long = row[LON]
            det_lat = row[LAT]
            det_weight = row[WEIGHT]
            weight_sum += det_weight
            det_heading = row[HEADING]
            new_cluster.roi_ids.append(det_id)
            sum_long += det_long * det_weight
            sum_lat += det_lat * det_weight
            sum_cos += cos(radians(det_heading)) * det_weight
            sum_sin += sin(radians(det_heading)) * det_weight
            sum_conf += det_confidence
        new_cluster.location.longitude = sum_long / weight_sum
        new_cluster.location.latitude = sum_lat / weight_sum
        # https://docs.python.org/3/library/math.html?highlight=atan2#math.atan2
        new_cluster.facing = normalized_heading(degrees(atan2(sum_sin / weight_sum, sum_cos / weight_sum)))
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
