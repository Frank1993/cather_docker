#!/usr/bin/env bash

# Building image from dockerfile.
docker build -f object_detection/retinanet/docker/Dockerfile --build-arg COMPONENT=traffic_signs_detector_retinanet -t telenav/traffic_signs_detector_retinanet .

# Run the container:
# sudo nvidia-docker run -v /data/:/data/ -v /home/flaviub/:/home/flaviub/ -v /home/adrianm/:/home/adrianm/ --net=host --name retinanet_mq -d -it telenav/traffic_signs_detector_retinanet
# sudo nvidia-docker run -v /data/:/data/ --net=host --name retinanet_mq -d -it telenav/traffic_signs_detector_retinanet
# sudo nvidia-docker run --name retinanet_mq -d -it telenav/traffic_signs_detector_retinanet
# sudo docker exec -it retinanet_mq /bin/bash
# sudo docker logs -f retinanet_mq