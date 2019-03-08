#!/usr/bin/env bash

set -e

PASSWD=$1

# build for dev
docker build -f mq_disk_serializer/docker/Dockerfile --build-arg COMPONENT=disk_serializer -t artefacts3.skobbler.net:8084/telenav/disk_serializer:1.1.0 .
docker push artefacts3.skobbler.net:8084/telenav/disk_serializer:1.1.0

# build for prod
docker login --username docker --password $PASSWD artefacts1.skobbler.net:8082
docker tag artefacts3.skobbler.net:8084/telenav/disk_serializer:1.1.0 artefacts1.skobbler.net:8082/telenav/disk_serializer:1.1.0
docker push artefacts1.skobbler.net:8082/telenav/disk_serializer:1.1.0
docker logout artefacts1.skobbler.net:8082
