#!/usr/bin/env bash

# Building image from dockerfile.
docker build -f mq_selectors/deploy/Dockerfile --build-arg COMPONENT=quality_orientation_image_selector -t telenav/quality_orientation_selector .
