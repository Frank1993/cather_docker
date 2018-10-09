import os
import signal
import subprocess
import sys
import time
import threading

sys.path.append(os.path.abspath('/home/job/apollo/imagerecognition/python_modules/'))
sys.path.append(os.path.abspath('/home/job/apollo/imagerecognition/python_modules/apollo_python_common/protobuf/'))

import argparse
import cv2
import json
import threading
from io import StringIO
import logging
from importlib import reload
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
import tensorflow as tf
import keras
from datetime import datetime
import classification.scripts.utils as utils

import apollo_python_common.image as image_api
import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
from apollo_python_common.io_utils import json_load
from apollo_python_common.mq.abstract_mq_consumer import AbstractMQConsumer

import apollo_python_common.proto_api as proto_api
from apollo_python_common.protobuf.classif_definitions_pb2 import *
from apollo_python_common.ml_pipeline.multi_threaded_predictor import RabbitMQProvider
from google.protobuf.json_format import MessageToJson

log_util.config("/home/job/apollo/imagerecognition/python_modules/image_orientation/run/correct_orientation.py")
logger = logging.getLogger(__name__)


class AllImagesProcessedExcepton(Exception):
    pass


class MQConsumer(AbstractMQConsumer):
    
    def __init__(self, config, total_count, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.total_count = total_count
        self.counter = 0
        self.output = StringIO()

    def consume_msg(self, message):
        image_proto = proto_api.read_image_proto(message.body)
        #print(image_proto)
        result = json.dumps(MessageToJson(image_proto))
        print(result)
        self.output.write(result + '\n')
        print(self.counter)

        self.counter += 1
        if self.counter >= self.total_count:
            message = 'All {} images are processed.'.format(str(self.counter))
            print(message)
            raise AllImagesProcessedExcepton(message)


def get_proto_for_path(path):
    return proto_api.get_new_image_proto("-1", -1, path, "US", 0,0, True )


def get_proto_for_paths(img_paths):
    return [get_proto_for_path(p) for p in img_paths]


def write_and_verify_output_file(output_file_path, data, output_lines_count):
    print('Writing results to {} ...'.format(output_file_path))
    with open(output_file_path, 'w') as output_file:
        output_file.write(print_consumer.output.getvalue())
        print_consumer.output.close()

    print('Writing complete. Verifying ...')
    with open(output_file_path, 'r') as output_file:
        line_counter = 0
        for line in output_file:
            line_counter += 1
        if output_lines_count != line_counter:
            raise Exception('Ouput detections count = {}. Expected = {}'.format(str(line_counter), str(output_lines_count)))

    print('Output file has expected {} detections.'.format(str(line_counter)))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagesZip',   help='Path to the zip file containg images', required=True, default=None)
    parser.add_argument('--imagesList',  help='Path to the file containing images in zip file', required=True, default=None)
    parser.add_argument('--outputDir',   help='Detection oputput path', required=True, default=None)
    parser.add_argument('--modelConfig', help='Path to the model config file', required=False, default=None)
    parser.add_argument('--logDir',      help='Log file', required=False, default=None)
    parser.add_argument('--dataDir',     help='Not used', required=False, default=None)
    parser.add_argument('--count',       help='Count', required=False, default=None)
    args = vars(parser.parse_args())

    print('Script arguments:')
    print('\t{}: {}'.format('imagesZip', args['imagesZip']))
    print('\t{}: {}'.format('imagesList', args['imagesList']))
    print('\t{}: {}'.format('outputDir', args['outputDir']))
    print('\t{}: {}'.format('modelConfig', args['modelConfig']))

    return args


def create_messages(args):
	images_zip_file = args['imagesZip']
	images_list_file = args['imagesList']
	with open(images_list_file) as f:
		img_paths = [os.path.join(images_zip_file + '@', line.rstrip('\n')) for line in f]

	return get_proto_for_paths(img_paths)


## Main function
if __name__== '__main__':
    args = parse_arguments()

    mq_rabbit_proc = subprocess.Popen(['rabbitmq-server', 'start'], stdout=subprocess.PIPE, preexec_fn=os.setsid)
    time.sleep(10)

    if args['modelConfig'] != None:
        object_detection_proc = subprocess.Popen(['sh', './start_object_detection_component.sh', args['modelConfig']], stdout=subprocess.PIPE, preexec_fn=os.setsid)
    else:
        object_detection_proc = subprocess.Popen(['sh', './start_object_detection_component.sh'], stdout=subprocess.PIPE, preexec_fn=os.setsid)
    time.sleep(10)

    img_proto_list = create_messages(args)
    if args['count'] == None:
        input_count = len(img_proto_list)
    else:
        input_count = int(args['count'])
        img_proto_list = img_proto_list[:input_count]
    print('Processing {} images ...'.format(str(input_count)))

    conf = io_utils.config_load("./mq_consumer_config.json")
    print_consumer = MQConsumer(conf, input_count)
    print_consumer.start()

    mq_provider = RabbitMQProvider("localhost", 5672, "guest", "guest")
    queue_name = "US_IMAGES"

    for img_proto in img_proto_list:
        mq_provider.send_message(queue_name, img_proto)

    print ("Waiting for all threads to finish ...")
    for t in threading.enumerate():
        try:
            t.join()
        except RuntimeError as err:
            continue
        except KeyboardInterrupt as err:
            break
    print("All threads are done.")

    output_file_path = os.path.join(args['outputDir'], 'detections.txt')
    write_and_verify_output_file(output_file_path, print_consumer.output, input_count)

    os.killpg(os.getpgid(mq_rabbit_proc.pid), signal.SIGTERM)

    print("Waiting RabbitMQ process to terminate ...")
    mq_rabbit_proc.wait()

    print('Waiting object detection process to terminate ...')
    try:
        object_detection_proc.wait()
    except KeyboardInterrupt as err:
        pass
