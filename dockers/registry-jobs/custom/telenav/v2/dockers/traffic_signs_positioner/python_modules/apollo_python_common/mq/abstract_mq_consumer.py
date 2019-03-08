import logging
from datetime import datetime
from collections import defaultdict

import apollo_python_common.audit as audit
import apollo_python_common.audit.ml_pipeline_audit as ml_pipeline_audit
import apollo_python_common.ml_pipeline.config_api as config_api
import apollo_python_common.proto_api as proto_api
from apollo_python_common.proto_api import MQ_Messsage_Type
from apollo_python_common.ml_pipeline.config_api import MQ_Param
from apollo_python_common.ml_pipeline.message_queue_providers import RabbitMQProvider, MESSAGE_PRIORITY_PROPERTY_KEY


class AbstractMQConsumer:

    def __init__(self, config, mq_message_type: MQ_Messsage_Type = MQ_Messsage_Type.IMAGE, **kwargs):
        self.config = config
        self.mq_message_type: MQ_Messsage_Type = mq_message_type
        self.logger = logging.getLogger(__name__)
        self.no_ack = config_api.get_config_param(MQ_Param.NO_ACK, config, True)  # Default without ACK
        self.mq_prefetch_count = config_api.get_config_param(MQ_Param.MQ_PREFETCH_COUNT, config, 30)
        self.mq_provider = RabbitMQProvider(config_api.get_config_param(MQ_Param.MQ_HOST, config),
                                            config_api.get_config_param(MQ_Param.MQ_PORT, config),
                                            config_api.get_config_param(MQ_Param.MQ_USERNAME, config),
                                            config_api.get_config_param(MQ_Param.MQ_PASSWORD, config))
        self.input_queue_name = config_api.get_config_param(MQ_Param.MQ_INPUT_QUEUE_NAME, config)
        self.input_errors_queue_name = config_api.get_config_param(MQ_Param.MQ_INPUT_ERRORS_QUEUE_NAME, config)
        audit.init(self.config)
        self.__audit_dict = defaultdict(lambda: dict())

    def set_audit_key_val(self, msg_id, key, val):
        self.__audit_dict[msg_id] = {key: val}

    def __clear_audit(self, msg_id):
        if msg_id in self.__audit_dict:
            del self.__audit_dict[msg_id]

    def consume_msg(self, message):
        pass

    def __consume_msg_with_time_tracking(self, input_message):
        start_time = datetime.now().timestamp()
        output_message = self.consume_msg(input_message)
        processing_time = round(datetime.now().timestamp() - start_time, 4)

        return processing_time,output_message

    def get_message_content(self, serialized_content):
        if self.mq_message_type == MQ_Messsage_Type.IMAGE:
            return proto_api.read_image_proto(serialized_content)
        elif self.mq_message_type == MQ_Messsage_Type.GEO_TILE:
            return proto_api.read_geotile_proto(serialized_content)
        else:
            raise Exception('Message type {} is not handled'.format(self.mq_message_type))

    def start(self):

        def on_mq_message(message):
            try:
                ml_pipeline_audit.one_message_was_received(self.get_message_content(message.body), self.config)
                processing_time,output_message = self.__consume_msg_with_time_tracking(message)
                ml_pipeline_audit.one_message_was_processed(self.get_message_content(output_message),
                                                            [("consume_msg", processing_time)], self.config,
                                                            self.__audit_dict[message.delivery_tag])

                if not self.no_ack:
                    self.mq_provider.acknowledge(message)

            except Exception as err:
                self.logger.exception(err)
                properties = self.get_message_properties_dict(message)
                self.mq_provider.send_message(self.input_errors_queue_name, message.body, properties)
                if not self.no_ack:
                    self.mq_provider.acknowledge(message)

            self.__clear_audit(message.delivery_tag)

        self.logger.info("Start listening for new input messages")

        try:
            self.mq_provider.start_consuming(self.input_queue_name,
                                             on_mq_message,
                                             no_ack=self.no_ack,
                                             mq_prefetch_count=self.mq_prefetch_count)
        except Exception as err:
            self.logger.exception(err)

    def send_message(self, dst_queue, body, properties=None):
        self.mq_provider.send_message(dst_queue, body, properties)

    def get_message_properties_dict(self, mq_message):
        return {MESSAGE_PRIORITY_PROPERTY_KEY: mq_message.priority}
