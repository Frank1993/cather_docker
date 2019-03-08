#!/bin/sh

CUDA_VISIBLE_DEVICES="0"
TRAIN_DATASET_CFG_JSON="./train_dataset_cfg.json"
TEST_DATASET_CFG_JSON="./test_dataset_cfg.json"
TRAIN_DATASET_GENERATOR_CFG_JSON="./train_dataset_generator_cfg.json"
TEST_DATASET_GENERATOR_CFG_JSON="./test_dataset_generator_cfg.json"

echo 'Parameters:'
echo 'TRAIN_DATASET_CFG_JSON = ' $TRAIN_DATASET_CFG_JSON
echo 'TEST_DATASET_CFG_JSON = ' $TEST_DATASET_CFG_JSON
echo 'TRAIN_DATASET_GENERATOR_CFG_JSON = ' $TRAIN_DATASET_GENERATOR_CFG_JSON
echo 'TEST_DATASET_GENERATOR_CFG_JSON = ' $TEST_DATASET_GENERATOR_CFG_JSON
echo 'CUDA_VISIBLE_DEVICES = ' $CUDA_VISIBLE_DEVICES

set -e
PYTHONPATH=../../../../:$PYTHONPATH
export PYTHONPATH
export TF_CPP_MIN_LOG_LEVEL=2.

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ../../../../classification/run/signs_facing_classifier/dataset_handler.py \
    --train_ds_cfg_json $TRAIN_DATASET_CFG_JSON \
    --test_ds_cfg_json $TEST_DATASET_CFG_JSON \
    --train_ds_gen_cfg_json $TRAIN_DATASET_GENERATOR_CFG_JSON \
    --test_ds_gen_cfg_json $TEST_DATASET_GENERATOR_CFG_JSON
    