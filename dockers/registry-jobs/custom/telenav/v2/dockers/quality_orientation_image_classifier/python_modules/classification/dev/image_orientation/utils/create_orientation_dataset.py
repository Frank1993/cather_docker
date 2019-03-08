import os
import sys
from tqdm import tqdm
import logging
import argparse
import PIL
from multiprocessing import Pool
import multiprocessing

import classification.scripts.utils as utils
import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util

rotate_dict = {
    "down": 180,
    "left": 90,
    "right": -90
}

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", help="source images folder", type=str, required=True)

    return parser.parse_args()


def get_base_folder(p):
    if p[-1] == "/":
        return os.path.dirname(p[:-1])
    else:
        return os.path.dirname(p)

def rotate_image(src_img_path):
    logger.info("Rotating : {}".format(os.path.basename(src_img_path)))
    img_name = os.path.basename(src_img_path)
    img = PIL.Image.open(src_img_path)

    for direction, deg_to_rotate in rotate_dict.items():
        direction_folder = os.path.join(base_folder,direction)
        io_utils.create_folder(direction_folder)

        dst_img_path = os.path.join(direction_folder,img_name)

        rotated_img = img.rotate(deg_to_rotate, expand=True)
        rotated_img.save(dst_img_path)

def create_orientation_dataset(input_folder):
    
    src_img_paths = utils.read_file_paths(input_folder)
    
    pool = Pool(multiprocessing.cpu_count() // 2)
        
    pool.map(rotate_image,src_img_paths)

if __name__ == '__main__':

    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    args = parse_arguments()
    
    
    base_folder = get_base_folder(args.input_folder)
    
    try:
        create_orientation_dataset(args.input_folder)
    except Exception as err:
        logger.error(err, exc_info=True)
        raise err
