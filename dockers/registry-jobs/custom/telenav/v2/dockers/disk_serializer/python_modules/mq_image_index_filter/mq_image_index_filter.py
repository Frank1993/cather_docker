import argparse
import logging
from collections import defaultdict

import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
import apollo_python_common.ml_pipeline.config_api as config_api
from apollo_python_common.ml_pipeline.config_api import MQ_Param
from apollo_python_common.mq.abstract_mq_consumer import AbstractMQConsumer


class Filter_MQ_Param(MQ_Param):
    FILTER_CONFIG_PATH = "filter_config_path"
    
class MQImageIndexFilter(AbstractMQConsumer):
    
    TRIP_ID_KEY = "trip_id"
    BOUND_LIST_KEY = "bound_list"
    START_KEY = "start"
    END_KEY = "end"
    FILTERS_KEY = "filters"
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.filters = self.__get_filters_dict(config_api.get_config_param(Filter_MQ_Param.FILTER_CONFIG_PATH, config,""))
        self.output_queue_name = config_api.get_config_param(MQ_Param.MQ_OUTPUT_QUEUE_NAME, config)

    def __get_filters_dict(self, filter_json_path):
        filter_json = io_utils.json_load(filter_json_path)
        filter_dict = defaultdict(list)
        filters = filter_json[self.FILTERS_KEY]
        for f in filters:
            trip_id = f[self.TRIP_ID_KEY]
            bound_list = f[self.BOUND_LIST_KEY]
            for bound_dict in bound_list:
                start,end = bound_dict[self.START_KEY], bound_dict[self.END_KEY]
                filter_dict[trip_id].append((start,end))

        return filter_dict
    
    def consume_msg(self, message):
        msg_proto = self.get_message_content(message.body)  
        output_message = msg_proto.SerializeToString()
        trip_id, img_index = int(msg_proto.metadata.trip_id), msg_proto.metadata.image_index
        
        valid_trip_id = trip_id in self.filters
            
        if not valid_trip_id:
            self.logger.info(f"Trip id {trip_id} not valid")
            return output_message
        
        valid_image_index = any([small_bound<=img_index<=big_bound for small_bound,big_bound in self.filters[trip_id]])
        
        if not valid_image_index:
            self.logger.info(f"Image index {img_index} from {trip_id} not in {self.filters[trip_id]}")
            return output_message
        
        self.logger.info(f"Valid: Trip Id = {trip_id} / Image_index = {img_index}")
        super().send_message(self.output_queue_name, output_message)
        return output_message
    
    
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", help="config file path", type=str, required=True)
    return parser.parse_args()

def run_serializer(conf_file):
    conf = io_utils.config_load(conf_file)
    serializer = MQImageIndexFilter(conf)
    serializer.start()

if __name__ == '__main__':
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    args = parse_arguments()
    run_serializer(args.config_file)

