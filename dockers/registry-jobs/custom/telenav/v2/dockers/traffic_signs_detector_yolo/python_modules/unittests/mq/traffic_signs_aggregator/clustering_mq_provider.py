import apollo_python_common.proto_api as proto_api
from unittests.mq.components.abstract_mq_provider import AbstractMQProvider


class ClusteringMQProvider(AbstractMQProvider):
    def __init__(self, config, imageset_data_path):
        self.imageset_data_path = imageset_data_path
        super().__init__(config)

    def get_proto_list(self):
        geotile_proto_list = []

        geotile_proto = proto_api.get_new_geotile_proto()
        geotile_proto.top_left.latitude = 43.00
        geotile_proto.top_left.longitude = 80.00
        geotile_proto.bottom_right.latitude = 44.00
        geotile_proto.bottom_right.longitude = 81.00
        input_image_set = proto_api.read_imageset_file(self.imageset_data_path)
        geotile_proto.image_set.CopyFrom(input_image_set)

        geotile_proto_list.append(geotile_proto.SerializeToString())
        return geotile_proto_list
