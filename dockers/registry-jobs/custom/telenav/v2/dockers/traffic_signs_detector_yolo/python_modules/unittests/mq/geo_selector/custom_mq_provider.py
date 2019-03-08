import apollo_python_common.proto_api as proto_api
from sklearn.utils import shuffle
from unittests.mq.components.abstract_mq_provider import AbstractMQProvider


class CustomMQProvider(AbstractMQProvider):

    def __init__(self, config, region_2_images):
        self.region_2_images = region_2_images
        super().__init__(config)

    def get_image_proto(self,region):
        return proto_api.get_new_image_proto("-1", -1, "/test/path/", region, -1, -1, True)
    
    def get_image_proto_list_from_region(self, region, nr_images):
        return [self.get_image_proto(region) for _ in range(nr_images)]
    
    def get_proto_list(self):
        all_proto_list = []
        for region, nr_images in self.region_2_images.items():
            region_proto_list = self.get_image_proto_list_from_region(region, nr_images)
            all_proto_list += region_proto_list

        return shuffle(all_proto_list)

