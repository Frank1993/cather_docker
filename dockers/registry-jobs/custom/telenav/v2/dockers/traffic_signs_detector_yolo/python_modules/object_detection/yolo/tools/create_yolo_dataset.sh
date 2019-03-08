#!/usr/bin/env bash

# You need to set this up first.
# path to the rois.bin file
ROI_TRAIN_PATH=/data/faces_license_plates_2018_09/train/rois.bin
ROI_TEST_PATH=/data/faces_license_plates_2018_09/test/rois.bin
# output model dir
MODEL_OUT_PATH=/data/flaviu/yolo_blur_test/
TRAIN_IMAGES_FILES=$MODEL_OUT_PATH/trainval/train_images_file.txt
VALIDATION_IMAGES_FILES=$MODEL_OUT_PATH/test/test_images_file.txt


YOLO_FOLDER=./../
cd $YOLO_FOLDER

WIDTH=1440
HEIGHT=1088
MIN_SIDE=2
CLASSES_IDS=./config/blur_classes_ids.json

PYTHONPATH="${PYTHONPATH}:../../:../../apollo_python_common/protobuf/:./"
export PYTHONPATH

mkdir -p MODEL_OUT_PATH

python3 tools/create_yolo_dataset.py --input_path $ROI_TRAIN_PATH \
        --output_path $MODEL_OUT_PATH/trainval/ \
        --images_file $TRAIN_IMAGES_FILES \
        --images_width $WIDTH \
        --images_height $HEIGHT \
        --min_side $MIN_SIDE \
        --classes_ids $CLASSES_IDS \
        --crop_vp False

python3 tools/create_yolo_dataset.py --input_path $ROI_TEST_PATH \
        --output_path $MODEL_OUT_PATH/test/ \
        --images_file $VALIDATION_IMAGES_FILES \
        --images_width $WIDTH \
        --images_height $HEIGHT \
        --min_side $MIN_SIDE \
        --classes_ids $CLASSES_IDS \
        --crop_vp False



