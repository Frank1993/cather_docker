import sys
import argparse
import logging

from apollo_python_common.ml_pipeline.multi_threaded_predictor import MultiThreadedPredictor
import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util

from sign_positioning.mq.sign_positioner import SignPositioner


class SignPositioningPredictor(MultiThreadedPredictor):
    '''
    Multi threaded predictor using first_local_app for signs positioning
    '''

    POSITIONING_ERROR_KEY = "PositioningError"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.sign_positioner = SignPositioner(config)

    def preprocess(self, image_proto):
        return image_proto

    def predict(self, image_proto_list):
        output_image_proto_list, es_messages = self.sign_positioner.process_image_proto_list(image_proto_list)
        for image_messages in es_messages:
                self.log_audit_warning(image_messages[0], image_messages[1])
        return output_image_proto_list

    def postprocess(self, localized_image_proto, image_proto):
        return localized_image_proto


def run_predictor(conf_file):
    conf = io_utils.config_load(conf_file)
    predictor = SignPositioningPredictor(conf)
    predictor.start()


def __parse_args(args):
    parser = argparse.ArgumentParser(description='Signs positioning.')
    parser.add_argument('--config_file',
                        help='Configuration file path',
                        default='../config/sign_positioning_config.json')
    return parser.parse_args(args)


if __name__ == '__main__':
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    # parse arguments
    args = sys.argv[1:]
    args = __parse_args(args)
    run_predictor(args.config_file)
