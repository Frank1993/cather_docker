import sys
import os
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../apollo_python_common/protobuf/'))
import apollo_python_common.proto_api as meta
import logging
from tqdm import tqdm
import apollo_python_common.log_util as log_util
from apollo_python_common.geometry.draw_util import draw_rois_to_file
import apollo_python_common.io_utils as io_utils


def main(conf):
    logger = logging.getLogger(__name__)
    io_utils.create_folder(conf.output_folder)
    logger.info("Metadata:")
    logger.info(meta.check_imageset(conf.rois_file))
    roi_dict = meta.create_images_dictionary(meta.read_imageset_file(conf.rois_file))
    for file_name, file_rois in tqdm(roi_dict.items()):
        path = os.path.join(conf.images_folder, file_name)
        if os.path.isfile(path):
            draw_rois_to_file(conf.images_folder, file_name, file_rois, conf.output_folder)
        else:
            print('File {} is missing'.format(path))


if __name__ == "__main__":
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    conf = io_utils.json_load('draw_rois.json')
    main(conf)
