import logging

from apollo_python_common.ml_pipeline.multi_threaded_predictor import RabbitMQProvider
import apollo_python_common.ml_pipeline.config_api as config_api
from apollo_python_common.ml_pipeline.config_api import MQ_Param


class AbstractMQProvider:
        
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.mq_provider = RabbitMQProvider(config_api.get_config_param(MQ_Param.MQ_HOST, config),
                                            config_api.get_config_param(MQ_Param.MQ_PORT, config),
                                            config_api.get_config_param(MQ_Param.MQ_USERNAME, config),
                                            config_api.get_config_param(MQ_Param.MQ_PASSWORD, config))
        self.mq_output_queue_name = config_api.get_config_param(MQ_Param.MQ_OUTPUT_QUEUE_NAME, config)
        self.proto_list = self.get_proto_list()
        
    def delete_queue(self, queue_name):
        self.mq_provider.delete_queue(queue_name)
    
    def get_proto_list(self):
        pass

    def get_proto_list_size(self):
        return len(self.proto_list)
    
    def start(self):
        for proto in self.proto_list:
            self.mq_provider.send_message(self.mq_output_queue_name, proto)
