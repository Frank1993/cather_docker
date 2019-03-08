import os

import apollo_python_common.proto_api as proto_api
from unittests.mq.components.abstract_mq_provider import AbstractMQProvider


class CustomMQProvider(AbstractMQProvider):

    def __init__(self, config, roi_file_path):
        self.roi_file_path = roi_file_path
        super().__init__(config)

    def get_proto_list(self):
        imageset = proto_api.read_imageset_file(self.roi_file_path)
        image_proto_list = []
        for ip in imageset.images:
            base_folder = os.path.dirname(self.roi_file_path)
            new_image_path = os.path.join(base_folder, os.path.basename(ip.metadata.image_path))
            ip.metadata.image_path = new_image_path
            image_proto_list.append(ip.SerializeToString())
        return image_proto_list
