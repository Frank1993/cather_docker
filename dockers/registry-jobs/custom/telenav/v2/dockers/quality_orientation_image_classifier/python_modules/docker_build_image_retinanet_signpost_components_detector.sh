#!/usr/bin/env bash

# Building image from dockerfile.
docker build -f object_detection/retinanet/docker/Dockerfile --build-arg COMPONENT=signpost_components_detector_retinanet -t telenav/sign_components_detector_retinanet .

# sudo nvidia-docker run -v /data/:/data/ --net=host --name retinanet_signpost_components_detector_mq -d -it telenav/sign_components_detector_retinanet
# sudo docker logs -f retinanet_signpost_components_detector_mq