#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES="0"

KERAS_MODEL='../tools/snapshots_101_v2/resnet101_traffic_signs_07.h5'
INFERENCE_MODEL='../tools/snapshots_101_v2/resnet101_traffic_signs_07.pb'
GENERATE_TENSORRT='1'
BACKBONE='resnet101'

echo 'Parameters:'
echo 'KERAS_MODEL:' $KERAS_MODEL
echo 'INFERENCE_MODEL:' $INFERENCE_MODEL
echo 'GENERATE_TENSORRT:' $GENERATE_TENSORRT
echo 'BACKBONE' $BACKBONE
echo '------------------------------------'

PYTHONPATH=../../../:../../../apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -u ./keras_file_to_pb_file.py \
    --keras_model $KERAS_MODEL \
    --inference_model $INFERENCE_MODEL \
    --generate_tensorrt $GENERATE_TENSORRT \
    --backbone $BACKBONE

