import argparse

import orbb_metadata_pb2
import apollo_python_common.io_utils as io_utils
import apollo_python_common.proto_api as proto_api
from object_detection.yolo.model.yolo_model import YoloModel

def generate_proto_data(folder):
    image_proto_list = []
    for img_file in io_utils.get_images_from_folder(folder):
        image_proto = proto_api.get_new_image_proto("-1", -1, img_file, "US",
                                                    40.715, -74.011, False)
        image_proto_list.append(image_proto)
    return image_proto_list


def test_yolo_model(folder):
    image_proto_list = generate_proto_data(folder)
    processed_image_list = []
    model = YoloModel(1472, 1120, "", "", None, 10, False)
    for image_proto in image_proto_list:
        pre_processed_data = model.pre_process(image_proto)
        processed_data = model.process([pre_processed_data])
        post_process_data = model.post_process(processed_data[0], image_proto)
        processed_image_list.append(post_process_data)
    return processed_image_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_path", type=str, required=True)
    parser.add_argument(
        "-o", "--output_path", type=str, required=False, default="./")
    args = parser.parse_args()
    image_proto_list = test_yolo_model(args.input_path)
    metadata = orbb_metadata_pb2.ImageSet()
    for image_proto in image_proto_list:
        image = metadata.images.add()
        image.CopyFrom(image_proto)
    proto_api.serialize_proto_instance(metadata, args.output_path)