import logging
import argparse
import os

import apollo_python_common.ml_pipeline.config_api as config_api
from apollo_python_common.ml_pipeline.config_api import MQ_Param
import apollo_python_common.log_util as log_util
import apollo_python_common.proto_api as proto_api
import apollo_python_common.io_utils as io_utils
from apollo_python_common.mq.abstract_mq_consumer import AbstractMQConsumer
from apollo_python_common.protobuf.classif_definitions_pb2 import *
from apollo_python_common.protobuf.orbb_definitions_pb2 import *

class QualityOrientationSelector(AbstractMQConsumer):
    IMAGE_QUALITY = "image_quality"
    IMAGE_ORIENTATION = "image_orientation"

    VALID_CLASS_NAMES = {
        IMAGE_QUALITY: "good",
        IMAGE_ORIENTATION: "up"
    }

    QUALITY_OUTCOME_GOOD = "good"
    QUALITY_OUTCOME_BAD = "bad"
    QUALITY_OUTCOME_ES_KEY = "quality_outcome"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.high_quality_queue_name = config_api.get_config_param(MQ_Param.HIGH_QUALITY_QUEUE_NAME, config)
        self.low_quality_queue_name = config_api.get_config_param(MQ_Param.LOW_QUALITY_QUEUE_NAME, config)

    def consume_msg(self, message):

        image_proto = proto_api.read_image_proto(message.body)
        is_high_quality_image = self.__is_high_quality_image(image_proto)

        if is_high_quality_image:
            output_queue_name, quality_outcome = self.high_quality_queue_name, self.QUALITY_OUTCOME_GOOD
        else:
            image_proto.processing_status = ProcessingStatus.Value("SKIP_AS_BAD")
            output_queue_name, quality_outcome = self.low_quality_queue_name, self.QUALITY_OUTCOME_BAD

        self.logger.info("Moving {} on queue {}".format(os.path.basename(image_proto.metadata.image_path),
                                                output_queue_name))

        output_message = image_proto.SerializeToString()
        properties = self.get_message_properties_dict(message)
        super().set_audit_key_val(message.delivery_tag, self.QUALITY_OUTCOME_ES_KEY, quality_outcome)
        super().send_message(output_queue_name, output_message, properties)
        
        return output_message

    def __is_classif_valid(self, image_proto, algorithm):
        classif_predictions = image_proto.features.classif_predictions
        classif_prediction = [cp for cp in classif_predictions if cp.algorithm == algorithm][0]

        return self.VALID_CLASS_NAMES[algorithm] == classif_prediction.chosen_pred_class_name

    def __is_high_quality_image(self, image_proto):
        valid_quality = self.__is_classif_valid(image_proto, self.IMAGE_QUALITY)
        valid_orientation = self.__is_classif_valid(image_proto, self.IMAGE_ORIENTATION)

        return valid_quality and valid_orientation


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", help="config file path", type=str, required=True)

    return parser.parse_args()


def run_predictor(conf_file):
    conf = io_utils.config_load(conf_file)
    predictor = QualityOrientationSelector(conf)
    predictor.start()


if __name__ == '__main__':
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    args = parse_arguments()
    run_predictor(args.config_file)
