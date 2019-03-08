#!/bin/sh


CUDA_VISIBLE_DEVICES="0"

echo 'Parameters:'
echo 'CUDA_VISIBLE_DEVICES = ' $CUDA_VISIBLE_DEVICES

set -e
PYTHONPATH=../../../:../../../ocr/attention_ocr/:../../../ocr/attention_ocr/datasets/$PYTHONPATH
export PYTHONPATH

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ../../attention_ocr/train.py \
    --dataset_name="fake_dataset_ts_smaller"\
    --train_log_dir='/home/docker/apollo/datasets/ocr_traffic_signs/weights/current_weights/'
#     --learning_rate=0.002\
#     --checkpoint='/home/docker/apollo/datasets/ocr_traffic_signs/weights/weights_dataset=1mil_v2/model.ckpt-58614'
   

    