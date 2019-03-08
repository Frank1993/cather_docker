import os
import tensorflow as tf
from keras import backend as K
from object_detection.keras_retinanet import models
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
import sys
import logging
import argparse
import apollo_python_common.log_util as log_util

'''
Transforms a network from Keras format to the Tensorflow inference format, optionally generating 
tensorrt FP16 and FP32 versions
'''

QUANTIZE = False
NUMBER_OF_OUTPUTS = 3


def parse_args(args):
    parser = argparse.ArgumentParser(description='Transforms a network from Keras format to the Tensorflow inference format')
    parser.add_argument("-k", "--keras_model_file", type=str, required=False,
                        help='The Keras (h5) model file.',
                        default='../tools/snapshots/resnet50_traffic_signs_01.h5')
    parser.add_argument("-i", "--inference_model", type=str, required=False,
                        help='The output tensorflow inference model file.',
                        default='../tools/snapshots/resnet50_traffic_signs_01.pb')
    parser.add_argument("-t", "--generate_tensorrt", type=bool, required=False,
                        help='Whether to generate tensorrt FP32 and FP16 versions',
                        default=True)
    parser.add_argument('--backbone', help='Backbone model used by retinanet.',
                        default='resnet50', type=str)
    return parser.parse_args(args)


def main(keras_model_file, backbone, inference_model, generate_tensorrt):
    logger = logging.getLogger(__name__)
    K.set_learning_phase(0)
    # net_model = keras.models.load_model(keras_model, custom_objects=custom_objects)
    net_model = models.load_model(keras_model_file, backbone_name=backbone, convert=True)
    sess = K.get_session()
    logger.info('net_model.output_names: {}'.format(list(net_model.output_names)))
    pred = [None] * NUMBER_OF_OUTPUTS
    pred_node_names = [None] * NUMBER_OF_OUTPUTS
    for i in range(NUMBER_OF_OUTPUTS):
        pred_node_names[i] = 'output' + str(i)
        pred[i] = tf.identity(net_model.outputs[i], name=pred_node_names[i])
    logger.info('output nodes names are: {}'.format(pred_node_names))
    if QUANTIZE:
        from tensorflow.tools.graph_transforms import TransformGraph
        transforms = ["sort_by_execution_order"]
        transformed_graph_def = TransformGraph(sess.graph.as_graph_def(), [], pred_node_names, transforms)
        constant_graph = graph_util.convert_variables_to_constants(sess, transformed_graph_def, pred_node_names)
    else:
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
    graph_io.write_graph(constant_graph, '', inference_model, as_text=False)
    logger.info('Saved the freezed graph (ready for inference) at: {}'.format(inference_model))
    if generate_tensorrt:
        from tensorflow.contrib import tensorrt as trt
        inference_file_name = os.path.splitext(os.path.basename(os.path.basename(inference_model)))[0]
        inference_model_folder = os.path.dirname(inference_model)

        fp16_graph = trt.create_inference_graph(
            input_graph_def = constant_graph,
            outputs = ['output0', 'output1', 'output2'],
            max_batch_size=1,
            max_workspace_size_bytes=4000000000,  # 4 GB
            precision_mode="FP16")  # Get optimized graph

        graph_io.write_graph(fp16_graph, '',
                             os.path.join(inference_model_folder, '{}_FP16.pb'.format(inference_file_name)),
                             as_text=False)

        fp32_graph = trt.create_inference_graph(
            input_graph_def = constant_graph,
            outputs = ['output0', 'output1', 'output2'],
            max_batch_size=1,
            max_workspace_size_bytes=4000000000,  # 4 GB
            precision_mode="FP32")  # Get optimized graph

        graph_io.write_graph(fp32_graph, '',
                             os.path.join(inference_model_folder, '{}_FP32.pb'.format(inference_file_name)),
                             as_text=False)


if __name__ == '__main__':
    log_util.config(__file__)
    # parse arguments
    args = sys.argv[1:]
    args = parse_args(args)
    main(args.keras_model_file, args.backbone, args.inference_model, args.generate_tensorrt)
