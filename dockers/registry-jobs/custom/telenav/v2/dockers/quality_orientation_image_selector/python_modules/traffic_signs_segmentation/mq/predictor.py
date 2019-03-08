import sys
import argparse
import logging

from apollo_python_common.ml_pipeline.multi_threaded_predictor import MultiThreadedPredictor
import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util

import traffic_signs_segmentation.utils.caffe_model as caffe_model


class CaffeNetPredictor(MultiThreadedPredictor):
    '''
    Multi threaded predictor using Caffe model
    '''
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.log_level = 0
        self.model = caffe_model.CaffeModel()

    def preprocess(self, image_proto):
        pre_processed_image = self.model.segmentation_pre_process(image_proto)
        return pre_processed_image

    def predict(self, caffe_image_data_list):
        caffe_image_data_list = self.model.segmentation_predict(caffe_image_data_list)
        return caffe_image_data_list

    def postprocess(self, one_file_predictions, image_proto):
        self.model.post_process(one_file_predictions, image_proto)
        return image_proto


def run_predictor(conf_file):
    conf = io_utils.json_load(conf_file)
    predictor = CaffeNetPredictor(conf)
    predictor.start()


def __parse_args(args):
    parser = argparse.ArgumentParser(description='Traffic signs detector. Takes images metadata from message queue.')
    parser.add_argument('--config_file',
                        help='Configuration file path',
                        default='../tools/caffe_config.json')
    return parser.parse_args(args)


if __name__ == '__main__':
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    # parse arguments
    args = sys.argv[1:]
    args = __parse_args(args)
    run_predictor(args.config_file)
