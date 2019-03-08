import argparse
import logging
import os

import folium
from folium.plugins import MarkerCluster
from tqdm import tqdm

import apollo_python_common.log_util as log_util
import apollo_python_common.proto_api as proto_api


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--input_clusters_file", type=str, required=True)
    parser.add_argument("-r", "--input_rois_file", type=str, required=True)
    parser.add_argument("-o", "--output_folder", type=str, required=True)
    return parser.parse_args()


def add_clusters(clusters, detections, view_map, output_folder='', draw_individual_clusters=False):
    logger = logging.getLogger(__name__)
    logger.info("Adding clusters to map")
    cluster_index = 0
    for cluster in tqdm(clusters.cluster):
        marker_cluster = MarkerCluster().add_to(view_map)

        for roi_id in cluster.roi_ids:
            trip_id, image_index, roi_type, lat, long, roi_x, roi_y = detections[roi_id]
            image_url = "http://openstreetcam.org/details/{}/{}".format(trip_id, image_index)
            html_str = '<a href="' + image_url + '"target="_blank">' + \
                       proto_api.get_roi_type_name(roi_type) + \
                       " roi_x=" + str(roi_x) + " roi_y=" + str(roi_y) + \
                       " roi_id=" + str(roi_id) + '</a>'
            popup = folium.Popup(html_str, max_width=300)
            folium.Marker(location=[lat, long], popup=popup).add_to(marker_cluster)

        if draw_individual_clusters:
            cluster_map = folium.Map()
            marker_cluster.add_to(cluster_map)
            cluster_index += 1
            map_name = "cluster{:03d}.html".format(cluster_index)
            cluster_file = os.path.join(output_folder, map_name)
            cluster_map.fit_bounds(marker_cluster.get_bounds())
            cluster_map.save(cluster_file)

    return view_map


if __name__ == "__main__":
    log_util.config(__file__)
    logger = logging.getLogger(__name__)

    args = get_args()
    input_clusters_file = args.input_clusters_file
    input_rois_file = args.input_rois_file
    output_folder = args.output_folder

    logger.info("Reading rois data...")
    input_image_set = proto_api.read_imageset_file(input_rois_file)
    logger.info("Reading cluster data...")
    clusters = proto_api.read_clusters_file(input_clusters_file)

    detections_dict = {roi.id: (image.metadata.trip_id,
                                image.metadata.image_index,
                                roi.type,
                                roi.local.position.latitude,
                                roi.local.position.longitude,
                                roi.rect.tl.col,
                                roi.rect.tl.row)
                       for image in input_image_set.images for roi in image.rois}

    map_osm = folium.Map()
    map_osm = add_clusters(clusters, detections_dict, map_osm, output_folder, draw_individual_clusters=False)

    output_file = os.path.join(output_folder, "cluster_map.html")
    map_osm.save(output_file)
