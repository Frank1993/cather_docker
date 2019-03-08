import os
import argparse

import apollo_python_common.proto_api as proto_api


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_roi_path", type=str, required=True)
    parser.add_argument("-o", "--output_roi_file_name", type=str, required=True)
    return parser.parse_args()


def convert_old_roi(old_file_path, new_file_name):
    old_proto_metadata = proto_api.read_imageset_file(old_file_path)
    for image in old_proto_metadata.images:
        image.metadata.id = '0'

    new_file_path = os.path.dirname(old_file_path)
    proto_api.serialize_proto_instance(old_proto_metadata, new_file_path, new_file_name)


if __name__ == "__main__":
    args = get_args()

    input_file_path = args.input_roi_path
    out_file_name = args.output_roi_file_name

    convert_old_roi(input_file_path, out_file_name)
