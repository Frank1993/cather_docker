FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:philly-ubuntu.16.04-cuda.10.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has tensorflow-gpu 2.0.0b0, in anaconda python36 with cuda 10.0, and cudnn 7" \
      repository="philly/jobs/custom/tensorflow" \
      tag="tf200b0-py36" \
      creator="joelpf" tooltype="tensorflow" \
      tooltypeversion="2.0.0b0" \
      createtime="6/26/2019"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get update && apt-get install -y --allow-change-held-packages --no-install-recommends \
         git \
         ca-certificates \
         libnccl2 \
         libnccl-dev \
         libjpeg-dev \
         libpng-dev &&\
         rm -rf /var/lib/apt/lists/*

# From baselines dockerfile
RUN apt-get -y update && apt-get -y install git wget python-dev python3-dev libopenmpi-dev python-pip zlib1g-dev cmake

ENV PYTHON_VERSION=3.6
ENV TENSORFLOW_VERSION=2.0.0b0

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda create -y --name tensorflow-py$PYTHON_VERSION python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl&& \
     /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/envs/tensorflow-py$PYTHON_VERSION/bin:$PATH

RUN pip install -q tensorflow-gpu==2.0.0-beta0

WORKDIR /workspace
RUN chmod -R a+w /workspace

