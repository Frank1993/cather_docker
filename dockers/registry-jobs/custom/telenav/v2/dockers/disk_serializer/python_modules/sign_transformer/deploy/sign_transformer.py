import logging
import argparse
import os
import apollo_python_common.ml_pipeline.config_api as config_api
import apollo_python_common.log_util as log_util
import apollo_python_common.proto_api as proto_api
import apollo_python_common.io_utils as io_utils
from apollo_python_common.mq.abstract_mq_consumer import AbstractMQConsumer


'''
Helper class to be used when the US model is used to detect similar signs from other regions.
'''
class SignTransformer(AbstractMQConsumer):
    REGIONS_TO_TRANSFORM = "regions_to_transform"
    FROM_SIGN_REGION_US = "from_sign_region_US"
    TO_SIGN_REGION = "to_sign_region_{}"
    DO_NOT_MAP = "_"
    MQ_OUTPUT_QUEUE_NAME_US = "mq_output_queue_name_US"
    MQ_OUTPUT_QUEUE_NAME_OTHERS = "mq_output_queue_name_OTHERS"
    OUT_QUEUE_ES_KEY = "output_queue"
    REGIONS_BASELINE = "regions_baseline"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.regions_to_transform = set(config_api.get_config_param(SignTransformer.REGIONS_TO_TRANSFORM, config, []))
        self.from_us_signs_list = config_api.get_config_param(SignTransformer.FROM_SIGN_REGION_US, config, [])
        self.to_regions_signs_dict_list = dict()
        for region in self.regions_to_transform:
            self.to_regions_signs_dict_list[region] = \
                config_api.get_config_param(SignTransformer.TO_SIGN_REGION.format(region), config, [])
        self.ts_map = self._get_traffic_signs_map()

    def _get_traffic_signs_map(self):
        ts_map = dict()
        for idx, us_sign in enumerate(self.from_us_signs_list):
            ts_map[us_sign] = dict()
            for target_region in self.to_regions_signs_dict_list.keys():
                cur_target_sign = self.to_regions_signs_dict_list[target_region][idx]
                if cur_target_sign != SignTransformer.DO_NOT_MAP:
                    ts_map[us_sign][target_region] = cur_target_sign
                else:
                    ts_map[us_sign][target_region] = None
        return ts_map

    def _get_output_queue_name(self, region):
        regions_baseline = config_api.get_config_param(SignTransformer.REGIONS_BASELINE, self.config, [])  # EU & CA
        if region in regions_baseline:
            return config_api.get_config_param(SignTransformer.MQ_OUTPUT_QUEUE_NAME_US, self.config)
        else:
            return config_api.get_config_param(SignTransformer.MQ_OUTPUT_QUEUE_NAME_OTHERS, self.config)

    def _add_mandatory_fields(self, image, roi):
        roi.local.facing = 0
        roi.local.position.latitude = image.sensor_data.raw_position.latitude
        roi.local.position.longitude = image.sensor_data.raw_position.longitude
        roi.local.distance = 0
        roi.local.angle_of_roi = 0
        roi.local.angle_from_center = 0

    def _filter_and_substitute_signs(self, input_image_proto, region):
        # filter or substitute the sign's types
        if region in self.regions_to_transform:
            self.logger.info("Processing image {}/{} from region {}".format(
                input_image_proto.metadata.trip_id, input_image_proto.metadata.image_index, region))
            for roi in list(input_image_proto.rois):
                # The images from target region will be just save din DB.
                # Mandatory fields have to be filled with default values.
                self._add_mandatory_fields(input_image_proto, roi)
                us_roi_type = proto_api.get_roi_type_name(roi.type)
                remove_roi = False
                if us_roi_type in self.from_us_signs_list:
                    target_type = self.ts_map[us_roi_type][region]
                    if target_type is not None:
                        # set the corespondent sign type for that region
                        roi.type = proto_api.get_roi_type_value(target_type)
                        self.logger.info("{} -> Convert {} -> {}".format(region, us_roi_type, target_type))
                    else:
                        remove_roi = True
                        self.logger.info("{} -> remove sign {}".format(region, us_roi_type))
                else:
                    remove_roi = True
                    self.logger.info("{} -> remove sign {}. Missing from map.".format(region, us_roi_type))

                if remove_roi:
                    # the sign type is not in the list of allowed signs for the target region -> remove it
                    input_image_proto.rois.remove(roi)

    def consume_msg(self, input_message):
        input_image_proto = proto_api.read_image_proto(input_message.body)
        region = input_image_proto.metadata.region
        output_queue_name = self._get_output_queue_name(region)
        self._filter_and_substitute_signs(input_image_proto, region)
        self.logger.info("Moving {} on queue {}".format(os.path.basename(input_image_proto.metadata.image_path),
                                                        output_queue_name))
        output_message = input_image_proto.SerializeToString()
        properties = self.get_message_properties_dict(input_message)
        super().set_audit_key_val(input_message.delivery_tag, self.OUT_QUEUE_ES_KEY, region)
        super().send_message(output_queue_name, output_message, properties)
        return output_message


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", help="config file path", type=str, required=True)
    return parser.parse_args()


def run_predictor(conf_file):
    conf = io_utils.config_load(conf_file)
    selector = SignTransformer(conf)
    selector.start()


if __name__ == '__main__':
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    args = parse_arguments()
    run_predictor(args.config_file)
