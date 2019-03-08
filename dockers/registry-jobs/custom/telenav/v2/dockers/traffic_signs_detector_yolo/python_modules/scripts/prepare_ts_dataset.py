import pandas as pd
import os
import sys
import time
from random import shuffle
from shutil import copyfile
from tqdm import tqdm
import logging
import argparse
from glob import glob
from classification.dev.image_orientation.utils.filter_bad_orientation_images import BadOrientationImageFilter
from classification.dev.image_quality.utils.filter_bad_quality_images import BadQualityImageFilter
import apollo_python_common.proto_api as proto_api
import apollo_python_common.log_util as log_util
import apollo_python_common.io_utils as io_utils
import keras

TRAIN_TEST_RATIO_OTHERS = 0.9
TRAIN_TEST_RATIO_TRIPS = 0.7
FTP_BUNDLE_PATH_BAD_ORIENTAION='/ORBB/data/image_orientation/good_bundle.zip'
FTP_BUNDLE_PATH_BAD_QUALITY='/ORBB/data/image_quality/good_bundle.zip'
global root_folder, exclude_from_train_file, raw_others_folder, raw_trips_folder, final_others_folder, final_trips_folder, logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--root_folder", type=str, required=True)
    parser.add_argument("-o", "--exclude_from_train_file", type=str, required=True)
    return parser.parse_args()


def exclude_pictures_with_issues():
    logger.info("Exclude pictures with issues (tagged in Canada, blurred etc)")
    df_to_exclude = pd.read_csv(exclude_from_train_file)
    dict_pict_to_exclude = dict()
    for idx, row in df_to_exclude.iterrows():
        dict_pict_to_exclude[(int(row.trip_id), int(row.seq_id))] = None

    roi_file_name = os.path.join(raw_others_folder, 'rois.bin')
    roi_meta = proto_api.read_imageset_file(roi_file_name)
    dict_images = {}
    idx = 0
    for image in tqdm(list(roi_meta.images)):
        dict_images[os.path.basename(image.metadata.image_path)] = (image.metadata.trip_id, image.metadata.image_index)
        if (int(image.metadata.trip_id), int(image.metadata.image_index)) in dict_pict_to_exclude:
            idx += 1
            roi_meta.images.remove(image)
    proto_api.serialize_proto_instance(roi_meta, raw_others_folder)


def split_train_test_others():
    logger.info("Splitting <others> dataset in train & test")
    # split in train 90% and test 10%, <others> dataset
    roi_file_name = os.path.join(raw_others_folder, 'rois.bin')
    roi_meta = proto_api.read_imageset_file(roi_file_name)
    others_images_list = list(roi_meta.images)
    shuffle(others_images_list)
    split_index = int(len(others_images_list) * TRAIN_TEST_RATIO_OTHERS)
    train_images = others_images_list[:split_index]
    test_images = others_images_list[split_index:]
    logger.info(('# All/Train/Test:', len(others_images_list), len(train_images), len(test_images)))
    # Train data
    io_utils.create_folder(os.path.join(final_others_folder, 'train'))
    train_imageset = proto_api.get_new_imageset_proto()
    for image in tqdm(train_images):
        __remove_invalid_rois(image)
        source_file_name = os.path.join(raw_others_folder, os.path.basename(image.metadata.image_path))
        dest_file_name = os.path.join(final_others_folder, 'train', os.path.basename(image.metadata.image_path))
        new_image = train_imageset.images.add()
        new_image.CopyFrom(image)
        copyfile(source_file_name, dest_file_name)
    proto_api.serialize_proto_instance(train_imageset, os.path.join(final_others_folder, 'train'))
    # Test data
    io_utils.create_folder(os.path.join(final_others_folder, 'test'))
    test_imageset = proto_api.get_new_imageset_proto()
    for image in tqdm(test_images):
        __remove_invalid_rois(image)
        source_file_name = os.path.join(raw_others_folder, os.path.basename(image.metadata.image_path))
        dest_file_name = os.path.join(final_others_folder, 'test', os.path.basename(image.metadata.image_path))
        new_image = test_imageset.images.add()
        new_image.CopyFrom(image)
        copyfile(source_file_name, dest_file_name)
    proto_api.serialize_proto_instance(test_imageset, os.path.join(final_others_folder, 'test'))


def __remove_invalid_rois(image):
    for roi in list(image.rois):
        if roi.type == 0:
            image.rois.remove(roi)


def split_train_test_trips():
    logger.info("Splitting <trips> dataset in train & test")
    # split in train 90% and test 10%, <trips> dataset
    roi_file_name = os.path.join(raw_trips_folder, 'rois.bin')
    roi_meta = proto_api.read_imageset_file(roi_file_name)
    dict_trips = dict()
    for image in roi_meta.images:
        if image.metadata.trip_id in dict_trips:
            dict_trips[image.metadata.trip_id].append(image)
        else:
            dict_trips[image.metadata.trip_id] = [image]
    all_trips = list(dict_trips.keys())
    shuffle(all_trips)
    split_index = int(len(all_trips) * TRAIN_TEST_RATIO_TRIPS)
    train_trips = all_trips[:split_index]
    test_trips = all_trips[split_index:]
    logger.info(('# All/Train/Test:', len(all_trips), len(train_trips), len(test_trips)))
    # Train data
    io_utils.create_folder(os.path.join(final_trips_folder, 'train'))
    train_imageset = proto_api.get_new_imageset_proto()
    for trip in train_trips:
        for image in dict_trips[trip]:
            __remove_invalid_rois(image)
            source_file_name = os.path.join(raw_trips_folder, os.path.basename(image.metadata.image_path))
            dest_file_name = os.path.join(final_trips_folder, 'train', os.path.basename(image.metadata.image_path))
            new_image = train_imageset.images.add()
            new_image.CopyFrom(image)
            copyfile(source_file_name, dest_file_name)
    proto_api.serialize_proto_instance(train_imageset, os.path.join(final_trips_folder, 'train'))
    # Test data
    io_utils.create_folder(os.path.join(final_trips_folder, 'test'))
    test_imageset = proto_api.get_new_imageset_proto()
    for trip in test_trips:
        for image in dict_trips[trip]:
            __remove_invalid_rois(image)
            source_file_name = os.path.join(raw_trips_folder, os.path.basename(image.metadata.image_path))
            dest_file_name = os.path.join(final_trips_folder, 'test', os.path.basename(image.metadata.image_path))
            new_image = test_imageset.images.add()
            new_image.CopyFrom(image)
            copyfile(source_file_name, dest_file_name)
    proto_api.serialize_proto_instance(test_imageset, os.path.join(final_trips_folder, 'test'))


def filter_bad_orientation_images():
    logger.info("Filter bad orientation images")
    # Filter bad orientation images
    keras.backend.clear_session()
    image_filter = BadOrientationImageFilter(FTP_BUNDLE_PATH_BAD_ORIENTAION, os.path.join(final_others_folder, 'train'), os.path.join(final_others_folder, 'train', 'bad_orientation'))
    image_filter.filter_bad_orietation_images()
    keras.backend.clear_session()
    image_filter = BadOrientationImageFilter(FTP_BUNDLE_PATH_BAD_ORIENTAION, os.path.join(final_others_folder, 'test'), os.path.join(final_others_folder, 'test', 'bad_orientation'))
    image_filter.filter_bad_orietation_images()
    keras.backend.clear_session()
    image_filter = BadOrientationImageFilter(FTP_BUNDLE_PATH_BAD_ORIENTAION, os.path.join(final_trips_folder, 'train'), os.path.join(final_trips_folder, 'train', 'bad_orientation'))
    image_filter.filter_bad_orietation_images()
    keras.backend.clear_session()
    image_filter = BadOrientationImageFilter(FTP_BUNDLE_PATH_BAD_ORIENTAION, os.path.join(final_trips_folder, 'test'), os.path.join(final_trips_folder, 'test', 'bad_orientation'))
    image_filter.filter_bad_orietation_images()


def filter_bad_quality_images():
    logger.info("Filter bad quality images")
    # Filter bad quality images
    keras.backend.clear_session()
    image_filter = BadQualityImageFilter(FTP_BUNDLE_PATH_BAD_QUALITY, os.path.join(final_others_folder, 'train'), os.path.join(final_others_folder, 'train', 'bad_quality'))
    image_filter.filter_bad_quality_images()
    keras.backend.clear_session()
    image_filter = BadQualityImageFilter(FTP_BUNDLE_PATH_BAD_QUALITY, os.path.join(final_others_folder, 'test'), os.path.join(final_others_folder, 'test', 'bad_quality'))
    image_filter.filter_bad_quality_images()
    keras.backend.clear_session()
    image_filter = BadQualityImageFilter(FTP_BUNDLE_PATH_BAD_QUALITY, os.path.join(final_trips_folder, 'train'), os.path.join(final_trips_folder, 'train', 'bad_quality'))
    image_filter.filter_bad_quality_images()
    keras.backend.clear_session()
    image_filter = BadQualityImageFilter(FTP_BUNDLE_PATH_BAD_QUALITY, os.path.join(final_trips_folder, 'test'), os.path.join(final_trips_folder, 'test', 'bad_quality'))
    image_filter.filter_bad_quality_images()


def merge_others_and_trips_protots():
    # Merge train proto
    train_proto = proto_api.add_missing_images(os.path.join(final_others_folder, 'train', 'rois.bin'), os.path.join(final_trips_folder, 'train', 'rois.bin'))
    proto_api.serialize_proto_instance(train_proto, os.path.join(root_folder, 'train'), 'rois_unclean')
    # Merge test proto
    test_proto = proto_api.add_missing_images(os.path.join(final_others_folder, 'test', 'rois.bin'), os.path.join(final_trips_folder, 'test', 'rois.bin'))
    proto_api.serialize_proto_instance(test_proto, os.path.join(root_folder, 'test'), 'rois_unclean')


def remove_entries_for_missing_pictures():
    logger.info("Removing entries for missing images")
    # Train
    count_removed = 0
    train_proto = proto_api.read_imageset_file(os.path.join(root_folder, 'train', 'rois_unclean.bin'))
    train_files = {os.path.basename(f):None for f in os.listdir(os.path.join(root_folder, 'train')) if os.path.isfile(os.path.join(os.path.join(root_folder, 'train', f)))}
    for image in tqdm(list(train_proto.images)):
        if os.path.basename(image.metadata.image_path) not in train_files:
            train_proto.images.remove(image)
            count_removed += 1
    proto_api.serialize_proto_instance(train_proto, os.path.join(root_folder, 'train'), 'rois')
    logger.info("{} images were removed".format(count_removed))
    # Test
    count_removed = 0
    test_proto = proto_api.read_imageset_file(os.path.join(root_folder, 'test', 'rois_unclean.bin'))
    test_files = {os.path.basename(f):None for f in os.listdir(os.path.join(root_folder, 'test')) if os.path.isfile(os.path.join(os.path.join(root_folder, 'test', f)))}
    for image in tqdm(list(test_proto.images)):
        if os.path.basename(image.metadata.image_path) not in test_files:
            test_proto.images.remove(image)
            count_removed += 1
    proto_api.serialize_proto_instance(test_proto, os.path.join(root_folder, 'test'), 'rois')
    logger.info("{} images were removed".format(count_removed))


def ensure_all_classes_in_test(train_proto):
    logger.info("Ensure all classes are in test")
    # Moving files from train to test set with the purpose to have at least three (3) instances of every sign type in test set.
    # This is for covering very rare classes.
    dict_sign_type_file = dict()
    for image in train_proto.images:
        for roi in image.rois:
            roi_type = proto_api.get_roi_type_name(roi.type)
            if roi_type not in dict_sign_type_file:
                dict_sign_type_file[roi_type] = list()
            dict_sign_type_file[roi_type].append(os.path.basename(image.metadata.image_path))

    res_train = proto_api.check_imageset(os.path.join(root_folder, 'train', 'rois.bin'))
    res_test = proto_api.check_imageset(os.path.join(root_folder, 'test', 'rois.bin'))
    to_move_from_train_to_test = list()
    for st, c in res_train.items():
        if res_test.get(st, 0) < 3 and c + res_test.get(st, 0) > 5:
            to_move_from_train_to_test.append((st, (3 - res_test.get(st, 0)), dict_sign_type_file[st][:(3 - res_test.get(st, 0))]))


def show_dataset_stats():
    res_train = proto_api.check_imageset(os.path.join(root_folder, 'train', 'rois.bin'))
    logger.info("   TRAIN:")
    for st, c in res_train.items():
        logger.info((st, c, res_train.get(st, 0)))
    res_test = proto_api.check_imageset(os.path.join(root_folder, 'test', 'rois.bin'))
    logger.info("   TEST:")
    for st, c in res_test.items():
        logger.info((st, c, res_test.get(st, 0)))


def remove_invalid_class_instances():
    logger.info("remove <INVALID> class instances")
    # removing rois having 'INVALID' class and image instances having missing files
    train_proto = proto_api.read_imageset_file(os.path.join(root_folder, 'train', 'rois.bin'))
    test_proto = proto_api.read_imageset_file(os.path.join(root_folder, 'test', 'rois.bin'))

    idx = 0
    for image in list(train_proto.images):
        image.metadata.image_path = os.path.basename(image.metadata.image_path)
        if not os.path.isfile(os.path.join(root_folder, 'train', image.metadata.image_path)):
            idx+=1
            train_proto.images.remove(image)
        for roi in list(image.rois):
            roi_type = proto_api.get_roi_type_name(roi.type)
            if roi_type == 'INVALID':
                image.rois.remove(roi)
                logger.info(('removed', roi.type))
    logger.info(('count invalid removed:', idx))

    idx = 0
    for image in test_proto.images:
        image.metadata.image_path = os.path.basename(image.metadata.image_path)
        if not os.path.isfile(os.path.join(root_folder, 'test', image.metadata.image_path)):
            idx+=1
            test_proto.images.remove(image)
        for roi in list(image.rois):
            roi_type = proto_api.get_roi_type_name(roi.type)
            if roi_type == 'INVALID':
                image.rois.remove(roi)
                logger.info(('count invalid removed:', roi.type))
    logger.info(('count', idx))

    proto_api.serialize_proto_instance(train_proto, os.path.join(root_folder, 'train'), 'rois')
    proto_api.serialize_proto_instance(test_proto, os.path.join(root_folder, 'test'), 'rois')


def __get_img_files_set(path):
    images_set = set()
    for file in set(glob(path + "*.jp*")):
        f = os.path.basename(file)
        images_set.add(os.path.join(path, f))
    return images_set


def __copy_images(img_source):
    logger.info("Copying {} files to final folder".format(img_source))
    io_utils.create_folder(final_others_folder)
    other_files = __get_img_files_set(os.path.join(final_others_folder, img_source))
    io_utils.create_folder(final_trips_folder)
    trips_files = __get_img_files_set(os.path.join(final_trips_folder, img_source))
    all_files = other_files.union(trips_files)
    for f in tqdm(all_files):
        copyfile(f, os.path.join(root_folder, img_source, os.path.basename(f)))


def copy_train_test_images():
    __copy_images("train")
    __copy_images("test")


if __name__ == "__main__":
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    args = get_args()
    root_folder = args.root_folder
    exclude_from_train_file = args.exclude_from_train_file
    raw_others_folder = os.path.join(root_folder, 'others_raw')
    raw_trips_folder = os.path.join(root_folder, 'trips_raw')
    final_others_folder = os.path.join(root_folder, 'others')
    final_trips_folder = os.path.join(root_folder, 'trips')

    exclude_pictures_with_issues()
    split_train_test_others()
    split_train_test_trips()
    filter_bad_quality_images()
    filter_bad_orientation_images()
    merge_others_and_trips_protots()
    copy_train_test_images()
    remove_entries_for_missing_pictures()

    train_proto = proto_api.read_imageset_file(os.path.join(root_folder, 'train', 'rois.bin'))
    test_proto = proto_api.read_imageset_file(os.path.join(root_folder, 'test', 'rois.bin'))
    logger.info(('# Train/Test:', len(train_proto.images), ',', len(test_proto.images)))
    show_dataset_stats()

    ensure_all_classes_in_test(train_proto)
    remove_invalid_class_instances()
    logger.info("FINAL STATISTICS:")
    show_dataset_stats()
    logger.info("All done")
