import sys
import argparse
import logging

from apollo_python_common.ml_pipeline.multi_threaded_predictor import MultiThreadedPredictor
import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util

import sign_positioning.utils.signs_positioning as signs_positioning


# C++ ErrorCodes:
NoProtoDimensionMap = 1
ProtoDimensionMapParseError = 2
ProtoDimensionMapMissingFields = 3
NoMatchedData = 4
NotFoundType = 5
NoDeviceName = 6
WriteError = 7
InvalidRoi = 8
EmptyImageSet = 9

class SignPositioningPredictor(MultiThreadedPredictor):
    '''
    Multi threaded predictor using first_local_app for signs positioning
    '''

    POSITIONING_ERROR_KEY = "PositioningError"

    def __init__(self, config, app_path, **kwargs):
        super().__init__(config, **kwargs)
        self.log_level = 0
        self.app_path = app_path
        self.signs_positioning_app = signs_positioning.SignsPositioning(self.app_path)

    def preprocess(self, image_proto):
        return image_proto

    def predict(self, image_proto_list):
        output_image_proto_list, return_code, error_data = self.signs_positioning_app.process(image_proto_list)
        self.handle_positioning_errors(return_code, error_data, output_image_proto_list)
        return output_image_proto_list

    def postprocess(self, localized_image_proto, image_proto):
        return localized_image_proto

    def handle_positioning_errors(self, return_code, error_data, image_proto_list):
        errors = [s.strip() for s in error_data[1].decode('utf-8').splitlines()][:-1]
        if 0 != return_code:
            raise Exception(errors)
        for error in errors:
            self.logger.warning(error)
            error_code_position = 1
            error_image_index_position = 3
            error_code = int(error.split(" ")[error_code_position])
            if error_code == NoDeviceName:
                image_index = int(error.split(" ")[error_image_index_position])
                super().set_audit_key_val(image_proto_list[image_index].metadata.image_path, self.POSITIONING_ERROR_KEY, error)


def run_predictor(conf_file, app_path):
    conf = io_utils.config_load(conf_file)
    predictor = SignPositioningPredictor(conf, app_path)
    predictor.start()


def __parse_args(args):
    parser = argparse.ArgumentParser(description='Signs positioning.')
    parser.add_argument('--config_file',
                        help='Configuration file path',
                        default='../tools/sign_positioning.json')
    parser.add_argument('--app_path',
                        help='first local app path',
                        default='../tools/first_local_app/')
    return parser.parse_args(args)


if __name__ == '__main__':
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    # parse arguments
    args = sys.argv[1:]
    args = __parse_args(args)
    run_predictor(args.config_file, args.app_path)
