import os
import argparse
import logging
from glob import glob

from apollo_python_common import log_util, proto_api, image, io_utils
from apollo_python_common.ml_pipeline.config_api import MQ_Param
from apollo_python_common.ml_pipeline.multi_threaded_predictor import RabbitMQProvider


class DS_MQ_Param(MQ_Param):
    INPUT_PATH = "input_path"
    OSC_API_URL = "osc_api_url"


class ProtoLoader:

    def __init__(self, config):
        self.mq_provider = RabbitMQProvider(config.mq_host, config.mq_port, config.mq_username, config.mq_password)
        self.queue = config.mq_input_queue_name
        self.error_queue = config.mq_input_errors_queue_name
        self.proto_file_list = ProtoLoader._get_proto_file_list(config.input_path)
        self.osc_api_url = config.osc_api_url

    @staticmethod
    def _get_proto_file_list(input_path):
        """ Gets all protobuf .bin file names from a given input folder. """
        logger.info("reading proto files from path {}...".format(input_path))
        return glob(os.path.join(input_path, "*.bin"))

    def _prepare_proto_list(self, proto_file):
        """
        Given a proto file, it returns the list of image protos from it after it updates the metadata.image_path to
        the OSC url.

        :param proto_file: the input proto file
        :return: a list of image protos
        """
        logger.info("preparing proto list from file {} ...".format(proto_file))

        image_set = proto_api.read_imageset_file(proto_file)
        proto_list = []
        error_proto_list = []

        for img_proto in image_set.images:
            img_path = image.get_image_path(img_proto.metadata.id, self.osc_api_url)
            if img_path is None:
                logger.info("could not find OSC image path for trip {} with image index {} ... skipping".format(
                    img_proto.metadata.trip_id, img_proto.metadata.image_index))
                error_proto_list.append(img_proto.SerializeToString())
            else:
                img_proto.metadata.image_path = img_path
                proto_list.append(img_proto.SerializeToString())

        return proto_list, error_proto_list

    def _send_proto_msg_to_queue(self, proto_list, queue):
        logger.info("sending {} proto image messages to {} ...".format(len(proto_list), queue))

        for img_proto in proto_list:
            self.mq_provider.send_message(queue, img_proto)

    def load_protos_in_queue(self):
        logger.info("loading proto messages from {} proto files...".format(len(self.proto_file_list)))

        for proto_file in self.proto_file_list:
            proto_list, error_proto_list = self._prepare_proto_list(proto_file)
            self._send_proto_msg_to_queue(proto_list, self.queue)
            self._send_proto_msg_to_queue(error_proto_list, self.error_queue)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", help="config file path", type=str, required=True)
    return parser.parse_args()


def run_loader(conf_file):
    config = io_utils.config_load(conf_file)
    proto_loader = ProtoLoader(config)
    proto_loader.load_protos_in_queue()


if __name__ == '__main__':
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    args = parse_arguments()
    run_loader(args.config_file)
