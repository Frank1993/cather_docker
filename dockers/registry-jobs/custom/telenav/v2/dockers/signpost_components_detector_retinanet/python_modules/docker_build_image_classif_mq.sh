#!/usr/bin/env bash

# Building image from dockerfile.
docker build -f classification/deploy/Dockerfile  --build-arg COMPONENT=quality_orientation -t telenav/classif_mq .
