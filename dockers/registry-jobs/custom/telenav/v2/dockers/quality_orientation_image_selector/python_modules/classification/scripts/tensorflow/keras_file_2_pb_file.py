import argparse
import logging
import os

import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
import classification.scripts.network as network
from keras import backend as K
from tensorflow.contrib import tensorrt as trt
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import graph_util


def construct_constant_graph(sess, outputs):
    return graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), outputs)
    
def save_regular_graph(constant_graph, output_folder):
    graph_io.write_graph(constant_graph, '', os.path.join(output_folder, 'model.pb'), as_text=False)

def save_tensorrt_graph(constant_graph, precision_mode, outputs, output_folder):
    trt_graph = trt.create_inference_graph(
            input_graph_def = constant_graph,
            outputs = outputs,
            max_batch_size=32,
            max_workspace_size_bytes=4000000000,  # 4 GB
            precision_mode=precision_mode)

    graph_io.write_graph(trt_graph, '',
                         os.path.join(output_folder, 'model_trt_{}.pb'.format(precision_mode)),
                         as_text=False)

def convert_2_pb(ftp_bundle_path, output_folder, sess):
    logger = logging.getLogger(__name__)
    io_utils.create_folder(output_folder)
    
    logger.info('Downloading model...')
    ssdish_model, params = network.load_model_bundle_from_ftp(ftp_bundle_path)
    model = network.load_end_2_end_model(params.conv_layer_name,
                                         params.img_size,
                                         ssdish_model,
                                        )

    outputs = ["model_2/dense_5/Softmax"]
    
    logger.info('Constructing constans graph...')
    constant_graph = construct_constant_graph(sess,outputs)

    logger.info('Exporting...')
    save_tensorrt_graph(constant_graph, "FP16", outputs, output_folder)
    save_tensorrt_graph(constant_graph, "FP32", outputs, output_folder)
    save_regular_graph(constant_graph, output_folder)                          
    
def configure_environment():
    K.set_learning_phase(0)
    sess = K.get_session()
    return sess
                         
                         
def parse_arguments():
    parser = argparse.ArgumentParser(description='Transforms a network from Keras format to the Tensorflow inference format')
    parser.add_argument("-k", "--ftp_bundle_path", type=str, required=False,
                        help='The Keras (h5) model file.')
    parser.add_argument("-i", "--output_folder", type=str, required=True)
    return parser.parse_args()



if __name__ == '__main__':
    log_util.config(__file__)
    args = parse_arguments()
    sess = configure_environment()
    convert_2_pb(args.ftp_bundle_path, args.output_folder, sess)
