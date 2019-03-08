import requests

import orbb_metadata_pb2
import orbb_definitions_pb2

import apollo_python_common.proto_api as proto_api

from sign_positioning.mq.combined_speed_limit_position_calculator import CombinedSpeedLimitPositionCalculator
from sign_positioning.mq.generic_sign_position_calculator import GenericSignPositionCalculator
from sign_positioning.mq.traffic_light_position_calculator import TrafficLightPositionCalculator
from sign_positioning.mq.combined_speed_limits_definition import COMBINED_SPEED_LIMITS


class SignPositioner:

    MISSING_MATCHED_DATA_ERROR_MSG = "Matched Data Error"
    INVALID_IMG_RES_MSG = "Image has invalid resolution"
    MISSING_DEVICE_TYPE_MSG = "Device type {} is missing"
    INVALID_ROI = "Roi {} is invalid"

    def __init__(self, config):
        self.phone_lenses_dict = proto_api.create_phone_lenses_dict(proto_api.read_phone_lenses(config.phone_lenses_proto))
        self.signs_dimensions_dict = proto_api.create_sign_dimensions_dict(proto_api.read_sign_dimensions(config.roi_dimensions_proto))
        self.warning_messages = list()
        self.osm_ways_url = config.osm_ways_url
        self.osm_timeout = config.osm_timeout
        self.vanishing_point_confidence = config.vanishing_point_confidence
        self.default_device_type = config.default_device_type

    def get_highway_type(self, image_proto):
        way_id = image_proto.match_data.matched_section.way_id
        request_url = "{}{}".format(self.osm_ways_url, way_id)
        try:
            response = requests.get(request_url, timeout=self.osm_timeout)
            response.raise_for_status()
            json_response = response.json()
            if 'tags' in json_response:
                for tag in json_response['tags']:
                    if tag['key'] == 'highway':
                        return tag['value']
        except:
            error_message = "Couldn't get highway type for way_id {}".format(way_id)
            self.add_warning_msg(image_proto, error_message)
            return ''

    def get_vanishing_point_x(self, image_proto):
        if image_proto.HasField("features") and image_proto.features.HasField("vanishing_point"):
            if image_proto.features.vanishing_point.confidence > self.vanishing_point_confidence:
                return image_proto.features.vanishing_point.vp.col

    def image_has_known_device(self, image_proto):
        return image_proto.sensor_data.device_type in self.phone_lenses_dict

    def get_sign_predictor_for_sign(self, roi_type, image_proto):
        if self.image_has_known_device(image_proto):
            phone_lens = self.phone_lenses_dict[image_proto.sensor_data.device_type]
        else:
            warning_message = self.MISSING_DEVICE_TYPE_MSG.format(image_proto.sensor_data.device_type)
            self.add_warning_msg(image_proto, warning_message)
            phone_lens = self.phone_lenses_dict[self.default_device_type]
        camera_location = image_proto.match_data.matched_position
        camera_heading = image_proto.match_data.matched_heading
        img_res = image_proto.sensor_data.img_res
        vp_x = self.get_vanishing_point_x(image_proto)
        if roi_type == orbb_definitions_pb2.TRAFFIC_LIGHTS_SIGN:
            predictor = TrafficLightPositionCalculator(phone_lens, camera_location, camera_heading, img_res, vp_x)
        elif roi_type in COMBINED_SPEED_LIMITS:
            other_rois = [roi for roi in image_proto.rois if roi.type is not roi_type]
            predictor = CombinedSpeedLimitPositionCalculator(phone_lens, camera_location, camera_heading, img_res, vp_x,
                                                             other_rois)
        else:
            predictor = GenericSignPositionCalculator(phone_lens, camera_location, camera_heading, img_res, vp_x)
        return predictor

    def process_image_proto_list(self, image_proto_list):
        self.clear_warning_msg()
        for image_proto in image_proto_list:
            if proto_api.empty_roi_list(image_proto):
                continue
            if not proto_api.image_has_valid_resolution(image_proto):
                raise Exception(self.INVALID_IMG_RES_MSG)
            if not proto_api.image_has_matched_position(image_proto):
                self.add_warning_msg(image_proto, self.MISSING_MATCHED_DATA_ERROR_MSG)
            highway_type = self.get_highway_type(image_proto)
            for roi in image_proto.rois:
                if not proto_api.valid_roi(roi):
                    self.add_warning_msg(self.INVALID_ROI.format(roi.id))
                predictor = self.get_sign_predictor_for_sign(roi.type, image_proto)
                obj_real_dimensions = self.signs_dimensions_dict[roi.type]
                sign_position = predictor.calculate_sign_position(roi, obj_real_dimensions, roi.local.angle_of_roi,
                                                                  highway_type)
                proto_api.add_sign_position_to_roi(sign_position, roi)
        return image_proto_list, self.warning_messages

    def add_warning_msg(self, image_proto, error_message):
        self.warning_messages.append((image_proto, error_message))

    def clear_warning_msg(self):
        self.warning_messages.clear()


def test_positioning(input_path, output_path, conf):
    input_metadata = proto_api.read_imageset_file(input_path)
    image_proto_list = [image for image in input_metadata.images]
    sign_positioner = SignPositioner(conf)
    processed_list = []
    for image_proto in image_proto_list:
        post_processed_image_proto = sign_positioner.process_image_proto_list([image_proto])
        processed_list.append(post_processed_image_proto)
    metadata = orbb_metadata_pb2.ImageSet()
    for image_proto, _ in processed_list:
        image = metadata.images.add()
        image.CopyFrom(image_proto[0])
    proto_api.serialize_proto_instance(metadata, output_path, "output")





