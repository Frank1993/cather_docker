#!/bin/sh


CUDA_VISIBLE_DEVICES="0"

echo 'Parameters:'
echo 'CUDA_VISIBLE_DEVICES = ' $CUDA_VISIBLE_DEVICES

set -e
PYTHONPATH=../../../:../../../ocr/attention_ocr/:../../../ocr/attention_ocr/datasets/$PYTHONPATH
export PYTHONPATH

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ../../attention_ocr/train.py \
    --dataset_name="fake_dataset"\
    --train_log_dir='/home/docker/apollo/datasets/ocr_sign_posts/weights/current_weights/'

#    --checkpoint=/tmp/attention_ocr/train_v2/model.ckpt-199067

    