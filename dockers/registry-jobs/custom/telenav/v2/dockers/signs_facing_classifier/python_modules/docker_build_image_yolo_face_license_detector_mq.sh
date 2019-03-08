#!/usr/bin/env bash

# Building image from dockerfile.
docker build -f object_detection/yolo/docker/Dockerfile --build-arg COMPONENT=face_license_detector_yolo -t telenav/face_license_detector_yolo .