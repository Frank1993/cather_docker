# Road Attribute Detection Caffe Image

## Project Goal
Detect various types of attributes in road street side and satellite imagery, including street signs, turn restrictions, lanes, poles, road width, etc.

> Current work focuses on detecting **street signs** in _flattened_ 360 degree street-side imagery.

## Contents
Modified version of Caffe toolkit required to build the CV model described in [Traffic Sign Detection and Classification in the Wild](https://cg.cs.tsinghua.edu.cn/traffic-sign/)

Together with some associated scripts used for run-time detection using model, and feature extraction given the detections.

### Models

No models are included in this image (they must be provided separately). However the scripts assume a particular output space for the models, defined in code/python//anno_func.py.

