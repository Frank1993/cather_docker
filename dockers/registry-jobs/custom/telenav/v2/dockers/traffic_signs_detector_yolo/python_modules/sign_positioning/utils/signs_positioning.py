import logging
import apollo_python_common.proto_api as proto_api
from sign_positioning.utils.first_local_runner import FirstLocalRunner
import apollo_python_common.log_util as log_util

TEST_INPUT_FILE = "/home/flaviub/positioning/rois.bin"
INPUT_FILE = "input"
OUTPUT_FILE = "output"



class SignsPositioning:

    def __init__(self, app_path):
        self.logger = logging.getLogger(__name__)
        self.app_path = app_path
        self.first_local_runner = FirstLocalRunner(app_path,
                                                   INPUT_FILE + ".bin",
                                                   OUTPUT_FILE + ".bin")

    def serialize_proto_message(self, image_proto_list):
        image_set = proto_api.get_new_imageset_proto()
        for image in image_proto_list:
            serialized_image = image_set.images.add()
            serialized_image.CopyFrom(image)
        proto_api.serialize_proto_instance(image_set, self.app_path, INPUT_FILE)

    def process_signs_positioning(self):
        return self.first_local_runner()

    def deserialize_proto_message(self):
        input_image_set = proto_api.read_imageset_file(self.app_path + OUTPUT_FILE + ".bin")
        image_proto_list = [image for image in input_image_set.images]
        return image_proto_list

    def process(self, image_proto_list):
        self.serialize_proto_message(image_proto_list)
        return_code, error_data = self.process_signs_positioning()
        output_proto_list = self.deserialize_proto_message()
        return output_proto_list, return_code, error_data

def test_sign_positioning():
    log_util.config(__file__)
    input_image_set = proto_api.read_imageset_file(TEST_INPUT_FILE)
    image_proto_list = [image for image in input_image_set.images]
    signs_positioning = SignsPositioning("./first_local_app")
    output_image_proto_list,_,_ = signs_positioning.process(image_proto_list)

if __name__ == '__main__':
    test_sign_positioning()
