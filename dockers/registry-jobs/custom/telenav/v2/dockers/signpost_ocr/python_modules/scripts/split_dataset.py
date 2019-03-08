import os
import logging
import random
import argparse
import shutil
from tqdm import tqdm
from collections import defaultdict

import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
import apollo_python_common.proto_api as meta


def extract_rois(rois, input_folder, output_folder, generate_metadata):
    logger.info("Extracting rois from {} to {}".format(input_folder, output_folder))

    result_dict = defaultdict(list)
    for full_path_filename, roi_list in tqdm(rois):
        src_file = os.path.join(input_folder, os.path.basename(full_path_filename))
        if os.path.isfile(src_file):
            dst_file = os.path.join(output_folder, os.path.basename(full_path_filename))
            shutil.copy(src_file, dst_file)
            result_dict[os.path.basename(full_path_filename)] = roi_list
        else:
            print('not found {}'.format(src_file))

    if generate_metadata:
        metadata = meta.create_imageset_from_dict(result_dict)
        meta.serialize_proto_instance(metadata, output_folder, 'rois')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images_folder", type=str, required=True)
    parser.add_argument("-o", "--images_out_folder", type=str, required=True)
    parser.add_argument("-s", "--split_ratio", type=float, default=0.8)
    parser.add_argument("-g", "--generate_metadata", type=bool, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    log_util.config(__file__)
    logger = logging.getLogger(__name__)

    args = get_args()
    images_folder = args.images_folder
    images_out_folder = args.images_out_folder
    split_ratio = args.split_ratio
    generate_metadata = args.generate_metadata

    rois_file = os.path.join(images_folder, 'rois.bin')
    train_folder = os.path.join(images_out_folder, 'train')
    valid_folder = os.path.join(images_out_folder, 'test')

    io_utils.create_folder(train_folder)
    io_utils.create_folder(valid_folder)

    roi_dict = meta.create_images_dictionary(meta.read_imageset_file(rois_file))
    items = list(roi_dict.items())
    random.shuffle(items)

    split_limit = int(len(items) * split_ratio)
    train_rois = items[:split_limit]
    valid_rois = items[split_limit:]

    extract_rois(train_rois, images_folder, train_folder, generate_metadata)
    extract_rois(valid_rois, images_folder, valid_folder, generate_metadata)
