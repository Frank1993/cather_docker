#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0

docker build -f traffic_signs_segmentation/docker/server/Dockerfile -t telenav/traffic_signs_segmentation_mq .

# Run the container:
# sudo nvidia-docker run --name traffic_signs_segmentation --net=host  --rm -ti telenav/traffic_signs_segmentation_mq