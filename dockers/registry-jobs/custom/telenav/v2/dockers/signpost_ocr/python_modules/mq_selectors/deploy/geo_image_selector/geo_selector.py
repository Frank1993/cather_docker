import logging
import argparse
import os

import apollo_python_common.ml_pipeline.config_api as config_api
from apollo_python_common.ml_pipeline.config_api import MQ_Param
import apollo_python_common.log_util as log_util
import apollo_python_common.proto_api as proto_api
import apollo_python_common.io_utils as io_utils
from apollo_python_common.mq.abstract_mq_consumer import AbstractMQConsumer
from apollo_python_common.protobuf.orbb_definitions_pb2 import *

class Geo_MQ_Param(MQ_Param):
    REGIONS_TO_PROCESS = "regions_to_process"
    REGIONS_TO_PROCESS_WITH_US_MODEL = "regions_to_process_with_us_model"
        
class GeoSelector(AbstractMQConsumer):
    GEO_REGION_ES_KEY = "geo_region"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.regions_to_process = set(config_api.get_config_param(Geo_MQ_Param.REGIONS_TO_PROCESS, config, ["US"]))
        self.default_output_queue = config_api.get_config_param(Geo_MQ_Param.MQ_OUTPUT_QUEUE_NAME, config)

        # temp solution in order to process images from other countries with US model
        self.regions_to_process_with_us_model = set(config_api.get_config_param(Geo_MQ_Param.REGIONS_TO_PROCESS_WITH_US_MODEL,
                                                                                config, []))
        
    def consume_msg(self, input_message):

        input_image_proto = proto_api.read_image_proto(input_message.body)
        region = input_image_proto.metadata.region

        # temp solution in order to process images from other countries with US model
        augmented_region = "US" if region in self.regions_to_process_with_us_model else region

        if augmented_region in self.regions_to_process:
            output_queue_name = f"{augmented_region}_IMAGES"
        else:
            output_queue_name = self.default_output_queue
            input_image_proto.processing_status = ProcessingStatus.Value("SKIP_AS_NO_MODEL_FOR_REGION")

        self.logger.info("Moving {} on queue {}".format(os.path.basename(input_image_proto.metadata.image_path),
                                                        output_queue_name))

        output_message = input_image_proto.SerializeToString()
        properties = self.get_message_properties_dict(input_message)
        super().set_audit_key_val(input_message.delivery_tag, self.GEO_REGION_ES_KEY, region)
        super().send_message(output_queue_name, output_message, properties)

        return output_message


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", help="config file path", type=str, required=True)

    return parser.parse_args()


def run_predictor(conf_file):
    conf = io_utils.config_load(conf_file)
    selector = GeoSelector(conf)
    selector.start()


if __name__ == '__main__':
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    args = parse_arguments()
    run_predictor(args.config_file)
