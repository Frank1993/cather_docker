import logging
import os
from collections import defaultdict

import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
import apollo_python_common.proto_api as proto_api
import orbb_definitions_pb2
from tqdm import tqdm


class RoiInvalidator:
    
    def __init__(self, input_rois_file_path, invalid_rois_folder, output_folder):
        self.input_rois_file_path = input_rois_file_path
        self.invalid_rois_folder = invalid_rois_folder
        self.output_folder = output_folder
  
    def __get_roi_id(self, roi):
        tl_row, tl_col = roi.rect.tl.row, roi.rect.tl.col
        br_row, br_col = roi.rect.br.row, roi.rect.br.col
        return f"{tl_row}-{tl_col}-{br_row}-{br_col}"
    

    def __build_invalid_rois_dict(self):
        img_name_2_rois = defaultdict(set)

        roi_img_names = os.listdir(self.invalid_rois_folder)
        for roi_img_name in roi_img_names:
            raw_img_name, extension = os.path.splitext(roi_img_name)[0],os.path.splitext(roi_img_name)[1]
            img_name, roi = raw_img_name.split("__")[0],raw_img_name.split("__")[1]

            full_img_name = f"{img_name}{extension}"
            img_name_2_rois[full_img_name].add(roi)
        
        return img_name_2_rois
    
    def __process_imageset(self,invalid_img_name_2_rois):
        
        imageset = proto_api.read_imageset_file(self.input_rois_file_path)
        for img_proto in tqdm(imageset.images):
            img_name = os.path.basename(img_proto.metadata.image_path)

            if img_name not in invalid_img_name_2_rois:
                continue

            invalid_rois = invalid_img_name_2_rois[img_name]

            for roi in img_proto.rois:
                roi_id = self.__get_roi_id(roi)
                if roi_id in invalid_rois:
                    print(f"Invalidating roi {roi_id} on image {img_proto.metadata.image_index} from trip {img_proto.metadata.trip_id}")
                    roi.validation = orbb_definitions_pb2.FALSE_POSITIVE

        return imageset
                    
    def invalidate_rois(self):        
        invalid_img_name_2_rois = self.__build_invalid_rois_dict()
        processed_imageset = self.__process_imageset(invalid_img_name_2_rois)
        proto_api.serialize_proto_instance(processed_imageset,self.output_folder,"validated_rois")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", help="config file path", type=str, required=True)

    return parser.parse_args()


def run_invalidator(conf_file):
    config = io_utils.json_load(conf_file)
    RoiInvalidator(config.input_rois_file_path,config.invalid_rois_folder,config.output_folder).invalidate_rois()

if __name__ == '__main__':
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    args = parse_arguments()
    run_invalidator(args.config_file)

        
        
   