#!/usr/bin/env bash

# Building image from dockerfile.
docker build -f mq_selectors/deploy/Dockerfile --build-arg COMPONENT=geo_image_selector -t telenav/geo_selector .
