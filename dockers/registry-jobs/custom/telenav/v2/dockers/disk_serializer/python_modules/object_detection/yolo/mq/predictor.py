import sys
import argparse
import logging
import threading

import apollo_python_common.log_util as log_util
import apollo_python_common.io_utils as io_utils

from apollo_python_common.ml_pipeline.multi_threaded_predictor import MultiThreadedPredictor
from object_detection.yolo.model.yolo_model import YoloPreProcessor
from object_detection.yolo.model.yolo_model import YoloProcessor
from object_detection.yolo.model.yolo_model import YoloPostProcessor


class YoloPredictor(MultiThreadedPredictor):
    '''
    Multi threaded predictor using Yolo model
    '''
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.log_level = 0
        self.config = config
        self.yolo_pre_processor_per_worker = dict()
        self.yolo_processor_per_worker = dict()
        self.yolo_post_processor = YoloPostProcessor(self.config)

    def get_pre_processor_per_worker(self):
        if threading.get_ident() not in self.yolo_pre_processor_per_worker:
            self.yolo_pre_processor_per_worker[threading.get_ident()] = YoloPreProcessor(self.config)
        return self.yolo_pre_processor_per_worker[threading.get_ident()]

    def get_processor_per_worker(self):
        if threading.get_ident() not in self.yolo_processor_per_worker:
            self.yolo_processor_per_worker[threading.get_ident()] = YoloProcessor(self.config)
        return self.yolo_processor_per_worker[threading.get_ident()]

    def preprocess(self, image_proto):
        yolo_pre_processor = self.get_pre_processor_per_worker()
        preprocessed_image = yolo_pre_processor.pre_process(image_proto)
        return preprocessed_image

    def predict(self, input_msg_list):
        yolo_processor = self.get_processor_per_worker()
        predictions_list = yolo_processor.process(input_msg_list)
        return predictions_list

    def postprocess(self, one_file_predictions, image_proto):
        image_proto = self.yolo_post_processor.post_process(one_file_predictions, image_proto)
        return image_proto


def run_predictor(conf_file):
    conf = io_utils.config_load(conf_file)
    predictor = YoloPredictor(conf)
    predictor.start()


def __parse_args(args):
    parser = argparse.ArgumentParser(description='Traffic signs detector. Takes images metadata from message queue.')
    parser.add_argument('--config_files',
                        help='Configuration file''s path',
                        default='../../config/yolo_mq.json')
    return parser.parse_args(args)


if __name__ == '__main__':
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    # parse arguments
    args = sys.argv[1:]
    args = __parse_args(args)
    run_predictor(args.config_files)
