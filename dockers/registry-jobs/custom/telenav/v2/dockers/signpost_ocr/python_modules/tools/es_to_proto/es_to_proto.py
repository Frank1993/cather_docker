import argparse
import elasticsearch.helpers as es_helpers
import logging
import sys
import tqdm


import apollo_python_common.audit as es
import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
import apollo_python_common.proto_api as proto_api
from tools.es_to_proto.constants import *


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_path", help="directory path for the output proto rois", type=str, required=True)
    parser.add_argument("-c", "--config_file", help="config file path", type=str, required=True)
    parser.add_argument("-q", "--query_file", help="file path for the query request", type=str, required=True)

    return parser.parse_args()


def add_rois_to_image(image, rois):
    for roi_dict in rois:
        new_roi = image.rois.add()
        new_roi.manual = roi_dict[MANUAL]
        new_roi.algorithm = roi_dict[ALGORITHM]
        new_roi.algorithm_version = roi_dict[ALGORITHM_VERSION]
        new_roi.type = proto_api.get_roi_type_value(roi_dict[TYPE])
        new_roi.rect.tl.col = roi_dict[RECT][TL][COL]
        new_roi.rect.tl.row = roi_dict[RECT][TL][ROW]
        new_roi.rect.br.col = roi_dict[RECT][BR][COL]
        new_roi.rect.br.row = roi_dict[RECT][BR][ROW]
        detections = roi_dict[DETECTIONS]
        for detection in detections:
            new_detection = new_roi.detections.add()
            new_detection.confidence = detection[CONFIDENCE]
            new_detection.type = proto_api.get_roi_type_value(detection[TYPE])


def add_metadata_to_image(image, metadata_dict):
    image.metadata.trip_id = metadata_dict[TRIP_ID]
    image.metadata.image_index = metadata_dict[IMAGE_INDEX]
    image.metadata.image_path = metadata_dict[IMAGE_PATH]
    image.metadata.region = metadata_dict[REGION]


def add_sensor_data_to_image(image, sensor_data_dict):
    image.sensor_data.speed_kmh = sensor_data_dict[SPEED]
    image.sensor_data.timestamp = int(sensor_data_dict[TIMESTAMP])
    image.sensor_data.raw_position.latitude = sensor_data_dict[RAW_POSITION][LATITUDE]
    image.sensor_data.raw_position.longitude = sensor_data_dict[RAW_POSITION][LONGITUDE]
    image.sensor_data.heading = sensor_data_dict[HEADING]


def add_matched_data_to_image(image, matched_data_dict):
    image.match_data.matched_position.latitude = matched_data_dict[MATCHED_POSITION][LATITUDE]
    image.match_data.matched_position.longitude = matched_data_dict[MATCHED_POSITION][LONGITUDE]
    image.match_data.matched_heading = matched_data_dict[MATCHED_HEADING]


def add_image_to_set(image_set, image_json_item):
    source = image_json_item[SOURCE]
    metadata = source[METADATA]
    sensor_data = source[SENSOR_DATA]
    new_image = image_set.images.add()
    add_metadata_to_image(new_image, metadata)
    add_sensor_data_to_image(new_image, sensor_data)
    if ROIS in source.keys():
        rois = source[ROIS]
        add_rois_to_image(new_image, rois)
    if MATCH_DATA in source.keys():
        matched_data = source[MATCH_DATA]
        add_matched_data_to_image(new_image, matched_data)


if __name__ == '__main__':
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    args = parse_arguments()

    config = io_utils.json_load(args.config_file)

    es.init(config)
    es_con = es.connection()
    if es_con is None:
        logger.error("Elasticsearch was not initialised. Cannot query audit data.")
        sys.exit(-1)
    index_name = es.audit_index_name()

    query = io_utils.json_load(args.query_file)
    scan_generator = es_helpers.scan(es_con, query=query, index=index_name)

    image_set = proto_api.get_new_imageset_proto()

    logger.info("Reading elasticsearch query")
    for image_item in tqdm.tqdm(scan_generator):
        add_image_to_set(image_set, image_item)

    if len(image_set.images) > 0:
        logger.info("Serializing the results to output. This could take a few minutes...")
        io_utils.create_folder(args.output_path)
        proto_api.serialize_proto_instance(image_set, args.output_path)
    else:
        logger.warning("Empty query result. Check query ")
