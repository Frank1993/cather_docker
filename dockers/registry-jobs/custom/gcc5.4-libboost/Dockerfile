# DockerFile to run executables for Machine-Translation group's training process

FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# --- Labels for the docker
LABEL description="This docker image supports gcc5.4 and libbost." \
      repository="philly/jobs/custom/generic-commandline-for-philly-submit-tool" \
      tag="v0.1" \
      creator="shujliu" tooltype="generic" \
      tooltypeversion="any" \
      createtime="3/15/2019"

WORKDIR /usr/local/

RUN apt-get update

RUN apt-get install -y build-essential cmake libboost-all-dev libgoogle-perftools-dev libprotobuf-dev libprotobuf9v5 libssl-dev openssl protobuf-compiler zlib1g-dev

WORKDIR /root/
COPY toolkit-execute /home/job/toolkit-execute
