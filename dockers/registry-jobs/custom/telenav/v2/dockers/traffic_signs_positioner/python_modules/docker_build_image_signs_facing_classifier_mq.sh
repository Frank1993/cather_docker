#!/usr/bin/env bash

# Building image from dockerfile.
docker build -f classification/deploy/Dockerfile  --build-arg COMPONENT=signs_facing_classifier -t telenav/signs_facing_classifier_mq .