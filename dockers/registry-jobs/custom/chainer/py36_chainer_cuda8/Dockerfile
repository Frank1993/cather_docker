FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.8.0-cudnn.6

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has 4.0.0b4 versions of cupy and chainer ubuntu 16.04 version of python36 with other toolkits" \
      repository="philly/jobs/custom/chainer" \
      tag="py36_chainer_cuda9" \
      creator="haerdoga" tooltype="chainer" \
      tooltypeversion="1.0" \
      createtime="2/23/2018"
# Everything above this line is required for the docker.

# the following is necessary for installing from the repo below
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    software-properties-common && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# this repo contains python3.6 for ubuntu 16.04
RUN add-apt-repository ppa:jonathonf/python-3.6

# install python 3.6 and other libraries from the repo
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3.6 \
    python3.6-dev \
    python3-pip \
    python3.6-venv \
    libsndfile-dev \
    libhdf5-dev && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# update pip since the repo does not have pip 3.6
# and pip install other required python libraries
RUN python3.6 -m pip install pip --upgrade && python3.6 -m pip install wheel && \
    pip3 install --upgrade setuptools && \
    pip3 install pandas h5py soundfile librosa mir_eval pathos pymongo monk resampy

RUN pip3 install cupy==4.0.0b4 chainer==4.0.0b4
