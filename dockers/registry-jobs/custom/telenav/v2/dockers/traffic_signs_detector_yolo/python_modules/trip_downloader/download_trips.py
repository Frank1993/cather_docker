import argparse
from datetime import datetime
import itertools
import logging
import os
import pandas as pd
from multiprocessing import Pool
import requests
import shutil
from sqlalchemy import text
import sys
import tqdm

import apollo_python_common.image as im_utils
import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
import apollo_python_common.proto_api as proto_api

from trip_downloader.constants import *
from trip_downloader.create_query import get_query
from trip_downloader.db_connection import DBConnection
from trip_downloader.parse_config import Config


class OSCDownloader(object):
    def __init__(self, download_path, config):
        self.download_path = download_path
        self.config = config

    @staticmethod
    def __create_image_url(image):
        storage_url = 'http://{}.openstreetcam.org/{}'
        storage = image.split('/')[0]
        image_url = storage_url.format(storage, image.replace(storage, ''))
        return image_url

    def __download_photo(self, image_data):
        logger = logging.getLogger(__name__)

        # only rois proto needed , skipping download of image
        if self.config.proto_rois_only:
            return True

        seq_id, image_index, image_location, shot_date = image_data
        image_url = self.__create_image_url(image_location)
        try:
            authentication = requests.auth.HTTPBasicAuth(self.config.osc_user, self.config.osc_password)
            request = requests.get(image_url, auth=authentication, stream=True, allow_redirects=True)
            if request.status_code == 200:
                file_name = os.path.basename(image_location)
                request.raw.decode_content = True
                with open("{}/{}".format(self.download_path, file_name), 'wb+') as f:
                    shutil.copyfileobj(request.raw, f)
            else:
                logger.info("Problem with the image request for {} : {}".format(image_url, request.reason))
        except Exception as ex:
            logger.info("Problem with the image request for {} : {}".format(image_url, ex))
            return False

        return True

    def __download_metadata(self, metadata):
        logger = logging.getLogger(__name__)
        metadata_url = self.__create_image_url(metadata)
        try:
            authentication = requests.auth.HTTPBasicAuth(self.config.osc_user, self.config.osc_password)
            request = requests.get(metadata_url, auth=authentication, stream=True, allow_redirects=True)
            if request.status_code == 200:
                file_path = "{}/{}".format(self.download_path, os.path.basename(metadata))
                request.raw.decode_content = True
                with open(file_path, 'wb+') as f:
                    shutil.copyfileobj(request.raw, f)
                return file_path
            else:
                logger.info("Problem with the metadata request for {} : {}".format(metadata_url, request.reason))
        except Exception as ex:
            logger.info("Problem with the metadata request for {} : {}".format(metadata_url, ex))
        return None

    def get_platform(self, seq_id):
        data_details = {'id': seq_id}
        try:
            details_response = requests.post(OSV_URL_DETAILS, data=data_details)
            metadata = details_response.json()[OSV][METADATA_FILENAME]
            meta_path = self.__download_metadata(metadata)
            with open(meta_path) as fis:
                device = fis.readline().split(';')[0]
            os.remove(meta_path)
        except Exception as ex:
            device = 'NA'
            logger.info("Problem getting the device for trip {} : {}".format(seq_id, ex))
        return device

    def __call__(self, image):
        success = self.__download_photo(image)
        return success


def get_trips_with_images(df):
    trips_with_images = df.groupby(SEQUENCE_ID)[IMAGE_INDEX].apply(set)
    return trips_with_images


def get_osc_trip_images(seq_id, image_ids=None):
    images = list()
    try:
        data = {'sequenceId': seq_id}
        seq_photos = requests.post(OSV_URL_PHOTO_LIST, data=data)
        osv_photos = seq_photos.json()[OSV][PHOTOS]
    except Exception as ex:
        logger.debug("Problem getting image list from OSC for trip {} : {}".format(seq_id, ex))
        return images
    for resp in osv_photos:
        sequence_index = int(resp[IMAGE_INDEX])
        shot_date = resp[SHOT_DATE]
        photo_timestamp = int(
            datetime.strptime(shot_date + " +0000", "%Y-%m-%d %H:%M:%S %z").timestamp()) if shot_date else 0

        if image_ids is None or sequence_index in image_ids:
            file_path = str(resp['name'].
                            replace('proc', 'ori').
                            replace('/th/', '/ori/').
                            replace('/lth/', '/ori/'))
            images.append((seq_id, sequence_index, file_path, photo_timestamp))
    return images


def download_images(df, download_path, config, threads_number):
    logger = logging.getLogger(__name__)

    trips_with_images = get_trips_with_images(df)

    logger.info("Getting the list of images")
    osc_images = list()
    for seq_id in tqdm.tqdm(trips_with_images.keys()):
        image_list = trips_with_images[seq_id]
        if config.full_trips:
            trip_images = get_osc_trip_images(seq_id)
        else:
            trip_images = get_osc_trip_images(seq_id, image_list)
        osc_images.extend(trip_images)

    if config.full_trips:
        for seq_id in config.trip_ids_included:
            if seq_id not in trips_with_images.keys():
                trip_images = get_osc_trip_images(seq_id)
                osc_images.extend(trip_images)

    osc_images_df = pd.DataFrame(osc_images, columns=[SEQUENCE_ID, IMAGE_INDEX, FILE_PATH, SHOT_DATE])
    merged_df = pd.merge(df, osc_images_df, how='outer')
    detections_df = merged_df[merged_df[LATITUDE].notnull()]
    empty_images_df = merged_df[merged_df[LATITUDE].isnull()]
    logger.info("Number of empty images: {}".format(empty_images_df.shape[0]))

    logger.info("Downloading images")
    pool = Pool(threads_number)
    osc_downloader = OSCDownloader(download_path, config)
    success_list = list(tqdm.tqdm(pool.imap(osc_downloader, osc_images), total=len(osc_images)))
    pool.close()
    fail_list = [not i for i in success_list]
    fail_to_download_images = list(itertools.compress(osc_images, fail_list))

    return detections_df, empty_images_df, fail_to_download_images


def add_sequence_device(sequence_devices, sequence_id, osc_downloader):
    if sequence_id not in sequence_devices:
        device_type = osc_downloader.get_platform(sequence_id)
        sequence_devices[sequence_id] = device_type
    return sequence_devices


def get_roi_position_from_row(row, im_width, im_height):
    x1 = row[ROI_X]
    y1 = row[ROI_Y]
    w1 = row[ROI_WIDTH]
    h1 = row[ROI_HEIGHT]
    x = x1 * im_width
    y = y1 * im_height
    w = w1 * im_width
    h = h1 * im_height
    return x, y, w, h


def add_rois_to_image_proto(image_proto, rois_df, im_width, im_height):
    sign_rois_df = rois_df[rois_df[PARENT_ID].isnull()]
    component_rois_df = rois_df[rois_df[PARENT_ID].notnull()]

    # iterate through the signs and add them to the image_proto before handling the components
    for _, row in sign_rois_df.iterrows():
        detection_mode = row[DETECTION_MODE]
        manual = detection_mode == MANUAL
        validation_status = row[VALIDATION_STATUS]

        x, y, w, h = get_roi_position_from_row(row, im_width, im_height)

        roi = image_proto.rois.add()
        roi.id = int(row[ROI_ID])
        roi.type = proto_api.get_roi_type_value(row[SIGN_TYPE])
        roi.rect.tl.row = int(max(0, y))
        roi.rect.tl.col = int(max(0, x))
        roi.rect.br.row = int(min(h + y, im_height))
        roi.rect.br.col = int(min(w + x, im_width))
        roi.manual = manual
        roi.validation = proto_api.get_roi_validation_value(validation_correspondence[validation_status])

        roi_detection = roi.detections.add()
        roi_detection.type = roi.type
        roi_detection.confidence = 1.0 if manual else row[CONFIDENCE]

    # handle the components now that we have their parents added to the proto
    for _, row in component_rois_df.iterrows():
        x, y, w, h = get_roi_position_from_row(row, im_width, im_height)
        parent_id = int(row[PARENT_ID])

        #find the parent roi and add the current component
        for parent_roi in image_proto.rois:
            if parent_id == parent_roi.id:
                new_component = parent_roi.components.add()
                new_component.box.tl.row = int(max(0, y))
                new_component.box.tl.col = int(max(0, x))
                new_component.box.br.row = int(min(h + y, im_height))
                new_component.box.br.col = int(min(w + x, im_width))

                new_component.type = proto_api.get_component_type_value(row[SIGN_TYPE])
                detection_mode = row[DETECTION_MODE]
                manual = detection_mode == MANUAL
                new_component.confidence = 1.0 if manual else row[CONFIDENCE]
                break


def create_proto_output(detections_df, empty_images_df, config, download_path, fail_to_download_images):
    logger = logging.getLogger(__name__)

    image_set = proto_api.get_new_imageset_proto()
    sequence_devices = dict()
    osc_downloader = OSCDownloader(download_path, config)

    grouped = detections_df.groupby(IMAGE_FIELDS)
    logger.info("Writing proto files")
    for (seq_id, seq_index, lon, lat, im_width, im_height, heading, region, file_path, shot_date), roi_df in tqdm.tqdm(grouped):
        # this image failed to download by some reason so we do not add it to the proto file
        if (seq_id, seq_index, file_path, shot_date) in fail_to_download_images:
            logger.info("Skipping image {} for failing to download".format(file_path))
            continue

        local_file_path = os.path.join(download_path, os.path.basename(file_path))
        if config.proto_rois_only:
            if im_height == 0 or im_width == 0:
                logger.info("Skipping image {} with height or width equal to 0".format(file_path))
                continue
        else:
            try:
                image_width, image_height = im_utils.get_size(local_file_path)
                if im_height == 0 or im_width == 0:
                    im_width = image_width
                    im_height = image_height
            except Exception as ex:
                logger.info("Skipping corrupt image {} : {} ".format(file_path, ex))
                continue

        image_proto = image_set.images.add()

        image_proto.sensor_data.raw_position.longitude = lon
        image_proto.sensor_data.raw_position.latitude = lat
        image_proto.sensor_data.heading = heading
        image_proto.sensor_data.img_res.width = int(im_width)
        image_proto.sensor_data.img_res.height = int(im_height)

        # TODO Add these fields when they are available
        image_proto.sensor_data.timestamp = int(shot_date)
        image_proto.sensor_data.speed_kmh = 0
        sequence_devices = add_sequence_device(sequence_devices, seq_id, osc_downloader)
        image_proto.sensor_data.device_type = sequence_devices[seq_id]

        image_proto.metadata.trip_id = str(seq_id)
        image_proto.metadata.image_index = seq_index
        image_proto.metadata.image_path = file_path
        image_proto.metadata.region = region
        image_proto.metadata.id = "0"

        add_rois_to_image_proto(image_proto, roi_df, im_width, im_height)

    for _, row in empty_images_df.iterrows():
        seq_id = row[SEQUENCE_ID]
        seq_index = row[IMAGE_INDEX]
        file_path = row[FILE_PATH]

        if (seq_id, seq_index, file_path) in fail_to_download_images:
            logger.info("Skipping empty image {} for failing to download".format(file_path))
            continue

        image_proto = image_set.images.add()
        image_proto.metadata.trip_id = str(seq_id)
        image_proto.metadata.image_index = seq_index
        image_proto.metadata.image_path = file_path
        image_proto.metadata.region = ""
        image_proto.metadata.id = "0"

    if config_data.remove_duplicates:
        image_set = proto_api.remove_duplicate_rois(image_set)

    proto_api.serialize_proto_instance(image_set, download_path)


def filter_detections_from_update_proto(df, image_set_update_proto, proto_rois_only):
    image_width_from_proto = 'image_width_from_proto'
    image_height_from_proto = 'image_height_from_proto'

    update_df = pd.DataFrame({IMAGE_PROTO: list(image_set_update_proto.images)})
    update_df.loc[:, SEQUENCE_ID] = update_df.loc[:, IMAGE_PROTO].apply(lambda im_proto: int(im_proto.metadata.trip_id))
    update_df.loc[:, IMAGE_INDEX] = update_df.loc[:, IMAGE_PROTO].apply(lambda im_proto: im_proto.metadata.image_index)
    update_df.loc[:, image_width_from_proto] = \
        update_df.loc[:, IMAGE_PROTO].apply(lambda im_proto: im_proto.sensor_data.img_res.width)
    update_df.loc[:, image_height_from_proto] = \
        update_df.loc[:, IMAGE_PROTO].apply(lambda im_proto: im_proto.sensor_data.img_res.height)
    update_df = update_df.drop([IMAGE_PROTO], axis=1)

    joined_df = pd.merge(update_df, df, how='left', on=[SEQUENCE_ID, IMAGE_INDEX])

    if proto_rois_only:
        joined_df.loc[joined_df[IMAGE_WIDTH] == 0, IMAGE_WIDTH] = joined_df[image_width_from_proto]
        joined_df.loc[joined_df[IMAGE_HEIGHT] == 0, IMAGE_HEIGHT] = joined_df[image_height_from_proto]

    joined_df = joined_df.drop([image_width_from_proto, image_height_from_proto], axis=1)

    return joined_df


def download_data(config, download_path, image_set_update_proto, threads_number):
    logger = logging.getLogger(__name__)

    # compose query based on config
    query = get_query(config)
    sql = text(query)

    # connect to DB and run query
    db_connection = DBConnection(config.db_user, config.db_password,
                                 config.db_host, config.db_port,
                                 config.db_name)
    df = pd.read_sql_query(sql, db_connection.get_session().get_bind())

    if image_set_update_proto is not None:
        df = filter_detections_from_update_proto(df, image_set_update_proto, config.proto_rois_only)

    # download images and keep list of those that failed
    detections_df, empty_images_df, fail_to_download_images = download_images(df, download_path, config, threads_number)

    # create and write proto output files
    create_proto_output(detections_df, empty_images_df, config, download_path, fail_to_download_images)


def config_is_valid(config):
    logger = logging.getLogger(__name__)

    if config.full_trips and config.trip_ids_included[0] == 'all':
        logger.error("You can not have all trips as full trips. "
                     "Either select some trip ids or remove the full trip flag")
        return False
    return True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type=str, required=True)
    parser.add_argument("-d", "--download_folder", type=str, required=True)
    parser.add_argument("-u", "--update_image_set_file", type=str, default=None)
    parser.add_argument("-t", "--threads_nr", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    log_util.config(__file__)
    logger = logging.getLogger(__name__)

    args = get_args()
    config_file = args.config_file
    download_folder = args.download_folder
    update_image_set_file = args.update_image_set_file
    threads_nr = args.threads_nr

    config_data = Config(config_file)
    if not config_is_valid(config_data):
        logger.error("Configuration is invalid!")
        sys.exit()

    io_utils.create_folder(download_folder)

    update_image_set_proto = proto_api.read_imageset_file(update_image_set_file) if update_image_set_file else None

    download_data(config_data, download_folder, update_image_set_proto, threads_nr)
