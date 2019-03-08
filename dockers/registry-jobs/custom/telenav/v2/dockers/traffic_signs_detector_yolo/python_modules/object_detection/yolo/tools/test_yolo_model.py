import argparse

import orbb_metadata_pb2
import apollo_python_common.io_utils as io_utils
import apollo_python_common.proto_api as proto_api
from object_detection.yolo.model.yolo_model import YoloPreProcessor
from object_detection.yolo.model.yolo_model import YoloProcessor
from object_detection.yolo.model.yolo_model import YoloPostProcessor

def generate_proto_data(folder):
    image_proto_list = []
    for img_file in io_utils.get_images_from_folder(folder):
        image_proto = proto_api.get_new_image_proto("-1", -1, img_file, "US",
                                                    40.715, -74.011, False)
        image_proto_list.append(image_proto)
    return image_proto_list


class YoloPredictorTest():
    def __init__(self, config):
        self.config = config
        self.yolo_pre_processor = YoloPreProcessor(self.config)
        self.yolo_processor = YoloProcessor(self.config)
        self.yolo_post_processor = YoloPostProcessor(self.config)

    def preprocess(self, image_proto):
        preprocessed_image = self.yolo_pre_processor.pre_process(image_proto)
        return preprocessed_image

    def predict(self, input_msg_list):
        predictions_list = self.yolo_processor.process(input_msg_list)
        return predictions_list

    def postprocess(self, one_file_predictions, image_proto):
        image_proto = self.yolo_post_processor.post_process(one_file_predictions, image_proto)
        return image_proto

def test_yolo_model(folder):
    conf = io_utils.config_load("./config/traffic_signs_detector_yolo_mq.json")
    predictor = YoloPredictorTest(conf)
    image_proto_list = generate_proto_data(folder)
    processed_image_list = []
    for image_proto in image_proto_list:
        pre_processed_data = predictor.preprocess(image_proto)
        processed_data = predictor.predict([pre_processed_data])
        post_processed_data = predictor.postprocess(processed_data[0], image_proto)
        processed_image_list.append(post_processed_data)
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