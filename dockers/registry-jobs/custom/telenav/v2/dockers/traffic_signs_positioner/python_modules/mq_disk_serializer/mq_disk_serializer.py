import os
import numpy as np
import shutil
import logging
from datetime import datetime
from multiprocessing import Pool
from time import sleep
import threading
import logging
import argparse

import apollo_python_common.proto_api as proto_api
import apollo_python_common.ml_pipeline.config_api as config_api
import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util

from apollo_python_common.proto_api import MQ_Messsage_Type
from apollo_python_common.mq.abstract_mq_consumer import AbstractMQConsumer
from apollo_python_common.ml_pipeline.config_api import MQ_Param

class DS_MQ_Param(MQ_Param):
    INPUT_PROTO_TYPE = "input_proto_type"
    OUTPUT_PATH = "output_path"
    TIMEOUT_PERIOD = "timeout_period"
    
class DiskSerializer(AbstractMQConsumer):
    
    IMAGESET_TYPE = "imageset"
    GEOTILE_TYPE = "geotile"
    
    def __init__(self, config, mq_message_type: MQ_Messsage_Type = MQ_Messsage_Type.IMAGE, **kwargs):
        super().__init__(config, mq_message_type=mq_message_type, **kwargs)
        self.config = config
        self.__check_valid_config()
        
        self.input_proto_type = config_api.get_config_param(DS_MQ_Param.INPUT_PROTO_TYPE, config, self.IMAGESET_TYPE)
        self.output_path = config_api.get_config_param(DS_MQ_Param.OUTPUT_PATH, config, "")
        self.timeout_period = config_api.get_config_param(DS_MQ_Param.TIMEOUT_PERIOD, config, 60)
        
        io_utils.create_folder(self.output_path)
        self.last_message_timestamp = None
        self.accumulated_protos = []
    
    def __check_valid_config(self):
        if self.config[DS_MQ_Param.INPUT_PROTO_TYPE] not in [DiskSerializer.IMAGESET_TYPE,DiskSerializer.GEOTILE_TYPE]:
            raise Exception(f"Wrong {DS_MQ_Param.INPUT_PROTO_TYPE} value")
        
        for key in [DS_MQ_Param.INPUT_PROTO_TYPE,DS_MQ_Param.OUTPUT_PATH,DS_MQ_Param.TIMEOUT_PERIOD]:
            if key not in self.config:
                raise Exception(f"Key {key} must exist in config")
    
    def __write_imageset_to_disk(self, batch_time):
        imageset = proto_api.get_new_imageset_proto()
        for image_proto in self.accumulated_protos:
            new_image_proto = imageset.images.add()
            new_image_proto.CopyFrom(image_proto)
        proto_api.serialize_proto_instance(imageset,self.output_path,f"batch_{batch_time}_rois")
        
    def __write_geotile_to_disk(self, batch_time):
        for index,geotile_proto in enumerate(self.accumulated_protos):
            proto_api.serialize_proto_instance(geotile_proto,self.output_path,f"batch_{batch_time}_geotile_{index}")
        
    def __flush_proto_to_disk(self):
        batch_time = datetime.utcnow().strftime('%Y-%m-%d_%H:%M:%S.%f')[:-3]
        if self.input_proto_type == self.IMAGESET_TYPE:
            self.__write_imageset_to_disk(batch_time)
            
        if self.input_proto_type == self.GEOTILE_TYPE:
            self.__write_geotile_to_disk(batch_time)
        
        self.accumulated_protos = []
            
    def __get_delta_timestamp(self):
        return (datetime.now() - self.last_message_timestamp).seconds
    
    def __timeout_thread(self):
        while True:
            sleep(1)
            delta_seconds = self.__get_delta_timestamp()
            logger.info(f"Waiting for {delta_seconds}s")
            if delta_seconds > self.timeout_period:
                if len(self.accumulated_protos)==0:
                    logger.info("Timeout exceeded. No protos accumulated...")
                else:
                    logger.info("Timeout exceeded. Writing to disk...")
                    self.__flush_proto_to_disk()
        
    def __start_timeout_thread(self):
        print("Started timeout thread...")
        timeout_thread = threading.Thread(target=self.__timeout_thread)
        timeout_thread.daemon = True
        timeout_thread.start()
    
    def __update_last_message_timestamp(self):
        self.last_message_timestamp = datetime.now()

    def consume_msg(self, message):
        msg_proto = self.get_message_content(message.body)        
        self.accumulated_protos.append(msg_proto)
        if self.last_message_timestamp is None:
            self.__start_timeout_thread()
        self.__update_last_message_timestamp()
        return msg_proto.SerializeToString()
    
    
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", help="config file path", type=str, required=True)
    return parser.parse_args()

def run_serializer(conf_file):
    conf = io_utils.config_load(conf_file)
    input_proto_type = config_api.get_config_param(DS_MQ_Param.INPUT_PROTO_TYPE, conf, DiskSerializer.IMAGESET_TYPE)
    message_type = MQ_Messsage_Type.IMAGE if input_proto_type == DiskSerializer.IMAGESET_TYPE else MQ_Messsage_Type.GEO_TILE
    serializer = DiskSerializer(conf,mq_message_type = message_type)
    serializer.start()

if __name__ == '__main__':
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    args = parse_arguments()
    run_serializer(args.config_file)

