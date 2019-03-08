from apollo_python_common import proto_api as proto_api
from scripts.traffic_signs_types import TrafficSignsTypes


class SignComponentsTypes(TrafficSignsTypes):

    def _get_name_to_caption(self, type_value):
        return proto_api.get_component_type_name(type_value)

    def caption_to_name(self, caption):
        return proto_api.get_component_type_value(caption)
