#!/usr/bin/env bash

# Building image from dockerfile.
docker build -f sign_clustering/docker/Dockerfile -t telenav/traffic_signs_aggregator .