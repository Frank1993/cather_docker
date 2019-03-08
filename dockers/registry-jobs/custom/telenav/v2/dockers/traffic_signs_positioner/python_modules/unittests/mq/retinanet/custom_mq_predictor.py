import os

import apollo_python_common.proto_api as proto_api
from unittests.mq.components.abstract_mq_provider import AbstractMQProvider


class CustomMQProvider(AbstractMQProvider):

    def __init__(self, config, dataset_path):

        self.dataset_path = dataset_path
        
        rois_path = os.path.join(dataset_path,"rois.bin")
        self.image_proto_list = proto_api.read_imageset_file(rois_path).images
        super().__init__(config)
        
    def get_proto_list(self):
        
        image_proto_list_ser = []

        for img_proto in self.image_proto_list:

            image_path = os.path.join(self.dataset_path, img_proto.metadata.image_path)
            new_proto = proto_api.get_new_image_proto("-1", -1, image_path, "US", -1, -1, True)
            image_proto_list_ser.append(new_proto)
        
        return image_proto_list_ser
    