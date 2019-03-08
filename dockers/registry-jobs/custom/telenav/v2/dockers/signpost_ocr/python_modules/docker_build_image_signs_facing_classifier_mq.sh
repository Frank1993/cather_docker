#!/usr/bin/env bash

# Building image from dockerfile.
# old component
# docker build -f classification/deploy/Dockerfile  --build-arg COMPONENT=signs_facing_classifier -t telenav/signs_facing_classifier_mq .

# signs facing 2.0
docker build -f classification/fast_ai/docker/Dockerfile --build-arg COMPONENT=signs_facing_classifier -t telenav/signs_facing_classifier_mq .