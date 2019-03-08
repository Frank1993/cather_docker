import argparse
import folium
from math import radians, cos, sin, asin, atan2, degrees
import pandas as pd
import random
from sklearn.metrics.cluster import adjusted_rand_score
import Geohash
from tqdm import tqdm

import apollo_python_common.io_utils as io_utils
import apollo_python_common.proto_api as proto_api
from apollo_python_common.map_geometry.geometry_utils import compute_haversine_distance
from sign_clustering import clustering
from sign_clustering.constants import *


GT_CLUSTER_ID = 'cluster_id'
DETECTION_CLASS = 'detection_class'


def filter_detections_by_geohash(rois_df, geohash_list):
    rois_df.loc[:, GEOHASH] = rois_df.progress_apply(lambda row: Geohash.encode(row[LATITUDE], row[LONGITUDE],
                                                                                precision=7), axis=1)
    filtered_df = rois_df[rois_df[GEOHASH].isin(geohash_list)]
    return filtered_df


def get_random_color():
    r = lambda: random.randint(0, 255)
    color = '#%02X%02X%02X' % (r(), r(), r())
    return color


def get_new_position(start_latitude, start_longitude, bearing, offset):
    start_longitude, start_latitude = map(radians, [start_longitude, start_latitude])
    bearing_rad = radians(bearing)
    r = 6371000  # Radius of earth in meters

    lat2 = asin(sin(start_latitude) * cos(offset / r) +
                cos(start_latitude) * sin(offset / r) * cos(bearing_rad))

    lon2 = start_longitude + atan2(sin(bearing_rad) * sin(offset / r) * cos(start_latitude),
                                   cos(offset / r) - sin(start_latitude) * sin(lat2))

    return degrees(lat2), degrees(lon2)


def draw_gt_clusters(df):
    grouped = df.groupby([GT_CLUSTER_ID])

    map_osm = folium.Map()
    folium.TileLayer('cartodbpositron').add_to(map_osm)
    idx_2_color = {i: get_random_color() for i in df[GT_CLUSTER_ID].unique()}

    for cluster_id, rois_df in grouped:
        for _, row in rois_df.iterrows():
            lat = row[LATITUDE]
            long = row[LONGITUDE]
            gt_cluster_id = row[GT_CLUSTER_ID]
            heading = row[FACING]
            url = row["url"]
            roi_type = row[TYPE]
            roi_x = row[ROI_X]
            roi_y = row[ROI_Y]
            html_str = '<a href="' + url + '"target="_blank">' + 'Cluster id : ' + str(gt_cluster_id) + ' ' + \
                       proto_api.get_roi_type_name(roi_type) + \
                       " roi_x=" + str(roi_x) + " roi_y=" + str(roi_y) + '</a>'
            popup = folium.Popup(html_str, max_width=300)
            circle_color = idx_2_color[gt_cluster_id]
            folium.CircleMarker([lat, long], radius=5, popup=popup, color=circle_color).add_to(map_osm)

            heading_head_lat, heading_head_lon = get_new_position(lat, long, heading, 5)
            folium.PolyLine(locations=[[lat, long], [heading_head_lat, heading_head_lon]],
                            weight=2, color='black').add_to(map_osm)

    map_osm.fit_bounds(map_osm.get_bounds())
    map_osm.save("gt_rois_map.html")


def draw_predicted_clusters(df):
    grouped = df.groupby([PREDICTED_CLUSTER_LABEL])

    rois_map = folium.Map()
    folium.TileLayer('cartodbpositron').add_to(rois_map)
    fn_rois_map = folium.Map()
    folium.TileLayer('cartodbpositron').add_to(fn_rois_map)
    fp_rois_map = folium.Map()
    folium.TileLayer('cartodbpositron').add_to(fp_rois_map)
    clusters_map = folium.Map()
    folium.TileLayer('cartodbpositron').add_to(clusters_map)
    idx_2_color = {i: get_random_color() for i in set(df[PREDICTED_CLUSTER_LABEL].values)}
    for pred_cluster_id, rois_df in grouped:
        for _, row in rois_df.iterrows():
            lat = row[LATITUDE]
            long = row[LONGITUDE]
            heading = row[FACING]
            url = row["url"] if "url" in row.keys() else ""
            detection_class = row[DETECTION_CLASS] if DETECTION_CLASS in row.keys() else ""
            roi_type = row[TYPE]
            roi_x = row[ROI_X]
            roi_y = row[ROI_Y]
            html_str = '<a href="' + url + '"target="_blank">' + 'Cluster id : ' + str(pred_cluster_id) + ' ' + \
                       proto_api.get_roi_type_name(roi_type) + \
                       " roi_x=" + str(roi_x) + " roi_y=" + str(roi_y) + '</a>'
            popup = folium.Popup(html_str, max_width=300)
            circle_color = idx_2_color[pred_cluster_id]
            if detection_class == 'FN':
                folium.CircleMarker([lat, long], radius=5, popup=popup, color=circle_color).add_to(fn_rois_map)
            elif detection_class == 'FP':
                folium.CircleMarker([lat, long], radius=5, popup=popup, color=circle_color).add_to(fp_rois_map)
            else:
                folium.CircleMarker([lat, long], radius=5, popup=popup, color=circle_color).add_to(rois_map)

            heading_head_lat, heading_head_lon = get_new_position(lat, long, heading, 5)
            folium.PolyLine(locations=[[lat, long], [heading_head_lat, heading_head_lon]],
                            weight=2, color='black').add_to(rois_map)

        cluster_lat, cluster_long = get_cluster_location(rois_df)
        cluster_popup = folium.Popup("Cluster id: " + str(pred_cluster_id), max_width=300)
        folium.Marker(location=[cluster_lat, cluster_long], popup=cluster_popup).add_to(clusters_map)

    rois_map.fit_bounds(rois_map.get_bounds())
    rois_map.save("predicted_rois_map.html")
    fn_rois_map.fit_bounds(fn_rois_map.get_bounds())
    fn_rois_map.save("false_negative_rois_map.html")
    fp_rois_map.fit_bounds(fp_rois_map.get_bounds())
    fp_rois_map.save("false_positive_rois_map.html")
    clusters_map.fit_bounds(clusters_map.get_bounds())
    clusters_map.save("predicted_clusters_map.html")


def get_cluster_location(cluster_rois_df, long_col_name=LONGITUDE, lat_col_name=LATITUDE):
    cluster_rois_df = cluster_rois_df.dropna(subset=[lat_col_name, long_col_name])
    latitude = (cluster_rois_df[lat_col_name] * cluster_rois_df[WEIGHT]).sum() / cluster_rois_df[WEIGHT].sum()
    longitude = (cluster_rois_df[long_col_name] * cluster_rois_df[WEIGHT]).sum() / cluster_rois_df[WEIGHT].sum()

    return latitude, longitude


def compare_clusters_to_gt(clusters_df, config):
    filtered_clusters_df = clusters_df.groupby([GT_CLUSTER_ID]).filter(lambda rois_df: len(rois_df) >= config.min_roi_samples_threshold).copy()
    filtered_clusters_df.loc[:, DETECTION_CLASS] = ''

    grouped_predicted_clusters = filtered_clusters_df.groupby([PREDICTED_CLUSTER_LABEL])
    grouped_predicted_clusters = sorted(list(grouped_predicted_clusters), key=lambda tuple: -len(tuple[1]))
    grouped_gt = filtered_clusters_df.groupby([GT_CLUSTER_ID])

    already_matched_clusters = []
    nr_rois_in_correct_clusters = 0
    nr_invalid_cluster_rois = 0
    distance_sum = 0

    for pred_cluster_id, pred_roi_ids_df in tqdm(grouped_predicted_clusters):
        if pred_cluster_id == INVALID_CLUSTER_ID:
            nr_invalid_cluster_rois = len(pred_roi_ids_df)
            filtered_clusters_df.loc[filtered_clusters_df[PREDICTED_CLUSTER_LABEL] == pred_cluster_id, DETECTION_CLASS] = 'FN'
            continue
        pred_rois_list = pred_roi_ids_df[ID].values

        max_matching_ids = 0
        matching_cluster_id = -1
        distance_between_matched_clusters = 0
        predicted_lat, predicted_long = get_cluster_location(pred_roi_ids_df)

        for gt_cluster_id, gt_roi_ids_df in grouped_gt:
            if gt_cluster_id in already_matched_clusters:
                continue

            gt_rois_list = gt_roi_ids_df[ID].values

            common_roi_ids = list(set(pred_rois_list) & set(gt_rois_list))
            nr_matching_ids = len(common_roi_ids)
            if nr_matching_ids > max_matching_ids:
                max_matching_ids = nr_matching_ids
                matching_cluster_id = gt_cluster_id
                column_names = list(gt_roi_ids_df.columns.values)
                if GT_LONGITUDE in column_names and GT_LATITUDE in column_names:
                    long_col_name, lat_col_name = GT_LONGITUDE, GT_LATITUDE
                else:
                    long_col_name, lat_col_name = LONGITUDE, LATITUDE
                gt_lat, gt_long = get_cluster_location(gt_roi_ids_df,
                                                       long_col_name=long_col_name,
                                                       lat_col_name=lat_col_name)
                distance_between_matched_clusters = compute_haversine_distance(predicted_long, predicted_lat,
                                                                               gt_long, gt_lat)

        if matching_cluster_id != -1:
            already_matched_clusters.append(matching_cluster_id)
            nr_rois_in_correct_clusters += max_matching_ids
            distance_sum += distance_between_matched_clusters
            filtered_clusters_df.loc[filtered_clusters_df[PREDICTED_CLUSTER_LABEL] == pred_cluster_id, DETECTION_CLASS] = 'TP'
        else:
            filtered_clusters_df.loc[filtered_clusters_df[PREDICTED_CLUSTER_LABEL] == pred_cluster_id, DETECTION_CLASS] = 'FP'

    true_positives = len(already_matched_clusters)
    false_positives = len(grouped_predicted_clusters) - len(already_matched_clusters)
    if nr_invalid_cluster_rois > 0:
        false_positives -= 1
    false_negatives = len(grouped_gt) - len(already_matched_clusters)

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    accuracy = true_positives / (true_positives + false_positives + false_negatives)

    distance_avg = distance_sum / len(already_matched_clusters)

    draw_gt_clusters(filtered_clusters_df)
    draw_predicted_clusters(filtered_clusters_df)

    return nr_rois_in_correct_clusters / (len(filtered_clusters_df) - nr_invalid_cluster_rois), \
           precision, recall, accuracy, distance_avg


def compare_cluster_proto_to_gt_file(clusters_proto, gt_file_path):
    gt_df = pd.read_csv(gt_file_path)
    clustered_rois_df = clustering.clusters_to_dataframe(clusters_proto)

    merged_df = pd.merge(gt_df, clustered_rois_df)
    completeness_score, precision, recall, accuracy, distance_avg = compare_clusters_to_gt(merged_df)

    return completeness_score, precision, recall, accuracy, distance_avg


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=str, required=True)
    parser.add_argument("-c", "--config_file", type=str, required=True)
    parser.add_argument("-g", "--ground_truth_file", type=str, required=True)
    parser.add_argument("-t", "--threads_nr", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    input_file = args.input_file
    config_file = args.config_file
    ground_truth_file = args.ground_truth_file
    nr_threads = args.threads_nr

    config = io_utils.config_load(config_file)

    rois_df = clustering.read_detections_input(input_file, config)
    gt_df = pd.read_csv(ground_truth_file)

    merged_df = pd.merge(gt_df, rois_df)

    merged_df = clustering.filter_detections_by_distance(merged_df, config.roi_distance_threshold)
    merged_df = clustering.filter_detections_by_gps_acc(merged_df, config.gps_accuracy_threshold)

    nr_clusters, clustered_df = clustering.get_clusters(merged_df, config, threads_number=nr_threads)
    compare_score = adjusted_rand_score(clustered_df[GT_CLUSTER_ID].values,
                                        clustered_df[PREDICTED_CLUSTER_LABEL].values)
    completeness_score, precision, recall, accuracy, distance_avg = compare_clusters_to_gt(clustered_df, config)
    print("Compare score: {}".format(compare_score))
    print("Completeness score: {}".format(completeness_score))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("Accuracy: {}".format(accuracy))
    print("Average distance between matched clusters: {}".format(distance_avg))
