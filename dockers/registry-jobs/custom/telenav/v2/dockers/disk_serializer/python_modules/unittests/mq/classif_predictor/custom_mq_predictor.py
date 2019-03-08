import os
from glob import glob

import apollo_python_common.proto_api as proto_api
from unittests.mq.components.abstract_mq_provider import AbstractMQProvider


class CustomMQProvider(AbstractMQProvider):

    def __init__(self, config, unittest_imgs_path):
        self.unittest_imgs_path = unittest_imgs_path
        super().__init__(config)

    def get_proto_list(self):

        classes = os.listdir(self.unittest_imgs_path)

        image_proto_list = []

        for class_name in classes:
            img_paths = glob(os.path.join(self.unittest_imgs_path, class_name, "*"))

            for img_path in img_paths:
                image_proto = proto_api.get_new_image_proto("-1", -1, img_path, "US", -1, -1, True)
                image_proto_list.append(image_proto)

        return image_proto_list
