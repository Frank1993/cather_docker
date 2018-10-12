import os
import signal
import subprocess
import sys
import time
import threading

sys.path.append(os.path.abspath('/home/job/apollo/imagerecognition/python_modules/'))
sys.path.append(os.path.abspath('/home/job/apollo/imagerecognition/python_modules/apollo_python_common/protobuf/'))

import logging
from importlib import reload
from glob import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import json
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
        self.results = []

    def consume_msg(self, message):
        image_proto = proto_api.read_image_proto(message.body)
        #print(image_proto)
        self.results.append(image_proto)
        print(self.counter)

        self.counter += 1
        if self.counter >= self.total_count:
            message = 'All {} images are processed.'.format(str(self.counter))
            print(message)
            raise AllImagesProcessedExcepton(message)

mq_rabbit_proc = subprocess.Popen(['rabbitmq-server', 'start'], preexec_fn=os.setsid)
time.sleep(10)

object_detection_proc = subprocess.Popen(['sh', './start_object_detection_component.sh'], preexec_fn=os.setsid)
time.sleep(10)

def get_proto_for_path(path):
    return proto_api.get_new_image_proto("-1", -1, path, "US", 0,0, True )

def get_proto_for_paths(img_paths):
    return [get_proto_for_path(p) for p in img_paths]

src_folder = "/home/job/apollo/demo_images"
img_paths = glob(src_folder + "/*")

img_proto_list = get_proto_for_paths(img_paths)
print(len(img_proto_list))

conf = io_utils.config_load("./mq_consumer_config.json")
print_consumer = MQConsumer(conf, len(img_paths))
print_consumer.start()

mq_provider = RabbitMQProvider("localhost", 5672, "guest", "guest")
queue_name = "US_IMAGES"

for img_proto in img_proto_list:
    mq_provider.send_message(queue_name, img_proto)

print ("Wait for threads ...")
for t in threading.enumerate():
    try:
        t.join()
    except RuntimeError as err:
        continue
    except KeyboardInterrupt:
        break
print ("All threads are done.")

os.killpg(os.getpgid(mq_rabbit_proc.pid), signal.SIGTERM)

print("Waiting RabbitMQ process to terminate ...")
mq_rabbit_proc.wait()

try:
    print('Waiting object detection process to terminate ...')
    object_detection_proc.wait()
except KeyboardInterrupt:
    pass

