FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has espnet with cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/espnet" \
      tag="espnet-py27-cuda9.0-cudnn7" \
      creator="t-yud" \
      createtime="12/07/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

# Set language
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Set timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

# Install packages
RUN apt-get update && apt-get install -y \
    wget \
    unzip

# wget espnet and kaldi
RUN cd / && \
    wget https://storage.sc1.philly.selfhost.corp.microsoft.com/txt2spch/t-yud/envs/espnet_cuda9.0_cudnn7.zip && \
    unzip espnet_cuda9.0_cudnn7.zip && mv espnet_cuda9.0_cudnn7 espnet && rm -rf espnet_cuda9.0_cudnn7.zip && \
    wget https://storage.sc1.philly.selfhost.corp.microsoft.com/txt2spch/t-yud/envs/kaldi_cuda9.0_cudnn7.zip && \
    unzip kaldi_cuda9.0_cudnn7.zip && mv kaldi_cuda9.0_cudnn7 kaldi && rm -rf kaldi_cuda9.0_cudnn7.zip
