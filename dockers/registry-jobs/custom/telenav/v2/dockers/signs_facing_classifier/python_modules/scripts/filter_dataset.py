import os
import logging
import json
import shutil
from tqdm import tqdm

import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
import apollo_python_common.proto_api as meta

if __name__ == "__main__":
    log_util.config(__file__)
    logger = logging.getLogger(__name__)

    with open('filter_dataset.json') as json_data_file:
        data = json.load(json_data_file)
    images_folder = data['IMAGES_FOLDER']
    needed_classes = [val for val in data['NEEDED_CLASSES']]
    images_out_folder = data['IMAGES_OUT_FOLDER']
    rois_file = os.path.join(images_folder, 'rois.bin')

    io_utils.create_folder(images_out_folder)
    roi_dict = meta.get_filtered_imageset_dict(rois_file, needed_classes)

    for file_base_name, rois in tqdm(roi_dict.items()):
        src_file = os.path.join(images_folder, file_base_name)
        dst_file = os.path.join(images_out_folder, file_base_name)
        shutil.copy(src_file, dst_file)

    metadata = meta.create_imageset_from_dict(roi_dict)
    meta.serialize_proto_instance(metadata, images_out_folder, 'rois')
    logger.info(meta.check_imageset(os.path.join(images_out_folder, 'rois.bin')))
