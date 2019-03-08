import argparse
import logging
from collections import defaultdict

import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
import apollo_python_common.proto_api as proto_api
from tqdm import tqdm


class RoiImageIndexFilter:
    
    TRIP_ID_KEY = "trip_id"
    BOUND_LIST_KEY = "bound_list"
    START_KEY = "start"
    END_KEY = "end"
    FILTERS_KEY = "filters"
    
    def __get_filters_dict(self, filter_json_path):
        filter_json = io_utils.json_load(filter_json_path)
        filter_dict = defaultdict(list)
        filters = filter_json[self.FILTERS_KEY]
        for f in filters:
            trip_id = f[self.TRIP_ID_KEY]
            bound_list = f[self.BOUND_LIST_KEY]
            for bound_dict in bound_list:
                start,end = bound_dict[self.START_KEY], bound_dict[self.END_KEY]
                filter_dict[trip_id].append((start,end))

        return filter_dict

    def __filter_invalid_image_indexes(self, imageset_proto, filters):
        to_delete_list = []
        for image_proto in tqdm(imageset_proto.images):
            trip_id = int(image_proto.metadata.trip_id)
            img_index = int(image_proto.metadata.image_index)

            if trip_id not in filters:
                continue

            bound_list = filters[trip_id]
            valid = any([small_bound<=img_index<=big_bound for small_bound,big_bound in bound_list])

            if not valid:
                print(f"Invalidated rois from trip_id = {trip_id} | image_index = {img_index}")
                print(f"Not in index bounds {bound_list} \n")
                to_delete_list.append(image_proto)


        for image_proto in to_delete_list:
            imageset_proto.images.remove(image_proto)
            
        return imageset_proto
        
    def filter_images(self,roi_path, filter_json_path, output_path):
        imageset = proto_api.read_imageset_file(roi_path)
        filters = self.__get_filters_dict(filter_json_path)
        
        processed_imageset = self.__filter_invalid_image_indexes(imageset, filters)
        
        proto_api.serialize_proto_instance(processed_imageset,output_path,"filtered_rois")
        
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", help="config file path", type=str, required=True)

    return parser.parse_args()


def run_filter(conf_file):
    config = io_utils.json_load(conf_file)
    RoiImageIndexFilter().filter_images(config.input_rois_file_path,
                                        config.filter_config_path,
                                        config.output_rois_folder)

if __name__ == '__main__':
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    args = parse_arguments()
    run_filter(args.config_file)
