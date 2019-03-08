import logging
import os
from tqdm import tqdm
import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
import apollo_python_common.proto_api as meta


def generate_meta(input_meta_file, filter_folders, output_meta_folder):
    input_meta_dict = meta.create_images_dictionary(meta.read_imageset_file(input_meta_file))
    out_meta_dict = dict()
    for full_file_name in tqdm(list(input_meta_dict.keys())):
        file_name = os.path.basename(full_file_name)
        file_exists = any([os.path.isfile(os.path.join(filter_folder, file_name)) for filter_folder in filter_folders])
        if file_exists:
            out_meta_dict[file_name] = input_meta_dict[file_name]

    meta_out = meta.create_imageset_from_dict(out_meta_dict)
    meta.serialize_proto_instance(meta_out, output_meta_folder)


if __name__ == "__main__":
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    # ********* PARAMETERS:
    conf = io_utils.json_load('filter_meta_from_folder.json')
    generate_meta(conf.input_meta_file, conf.filter_folders, conf.output_meta_folder)
