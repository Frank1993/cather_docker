import os
import cv2
from tqdm import tqdm
import logging
import argparse
import PIL
from multiprocessing import Pool
import multiprocessing

import apollo_python_common.proto_api as proto_api
import apollo_python_common.image as image_api
import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util


class QuickTagRoiDatasetGenerator:

    def __init__(self, src_folder_path, output_path):
        self.src_folder_path = src_folder_path
        self.output_path = output_path
        io_utils.create_folder(output_path)
        
    def __draw_roi_on_img(self, img, roi):
        tl_row = roi.rect.tl.row
        tl_col = roi.rect.tl.col
        br_row = roi.rect.br.row
        br_col = roi.rect.br.col

        return cv2.rectangle(img.copy(), (tl_col, tl_row), (br_col, br_row), (255,0,0), 6)
    
    def __get_roi_img_name(self,img_name, roi):
        raw_img_name, extension = os.path.splitext(img_name)[0],os.path.splitext(img_name)[1]
        tl_row, tl_col = roi.rect.tl.row, roi.rect.tl.col
        br_row, br_col = roi.rect.br.row, roi.rect.br.col
        return f"{raw_img_name}__{tl_row}-{tl_col}-{br_row}-{br_col}{extension}"
    
    def __save_roi_img(self, roi_img, roi, img_name):
        roi_img_name = self.__get_roi_img_name(img_name,roi)
        roi_img_path = os.path.join(self.output_path,roi_img_name)
        PIL.Image.fromarray(roi_img).save(roi_img_path)
        
    def process_image(self, image_proto):
        img_name = os.path.basename(image_proto.metadata.image_path)
        img_path = os.path.join(self.src_folder_path,img_name)

        orig_image = image_api.get_rgb(img_path)

        for roi in tqdm(image_proto.rois):
            roi_img = self.__draw_roi_on_img(orig_image,roi)
            self.__save_roi_img(roi_img,roi,img_name)

    def generate_dataset(self):
        
        imageset = proto_api.read_imageset_file(os.path.join(self.src_folder_path,"rois.bin"))
        image_proto_list = imageset.images
        
        nr_threads = multiprocessing.cpu_count() // 2
        Pool(nr_threads).map(self.process_image,image_proto_list)
        
        
        
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", help="config file path", type=str, required=True)

    return parser.parse_args()


def run_generator(conf_file):
    config = io_utils.json_load(conf_file)
    generator = QuickTagRoiDatasetGenerator(config.src_folder_path, config.output_path)
    generator.generate_dataset()
    

if __name__ == '__main__':
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    args = parse_arguments()
    run_generator(args.config_file)