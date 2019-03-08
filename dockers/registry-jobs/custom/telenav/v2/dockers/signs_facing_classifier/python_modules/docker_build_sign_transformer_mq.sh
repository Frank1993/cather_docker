#!/usr/bin/env bash

# Building image from dockerfile.
docker build -f sign_transformer/deploy/Dockerfile -t telenav/sign_transformer .

# Run the container:
# sudo docker run -v /data/:/data/ -v /home/flaviub/:/home/flaviub/ -v /home/adrianm/:/home/adrianm/ --net=host --name sign_transformer -d -it telenav/sign_transformer
# sudo docker run -v /data/:/data/ --net=host --name sign_transformer -d -it telenav/sign_transformer
# sudo docker exec -it sign_transformer /bin/bash
# sudo docker logs -f sign_transformer
#docker run -v /Users/adrianm:/Users/adrianm --net=host --name sign_transformer -d -it telenav/sign_transformer