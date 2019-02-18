FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.8.0-cudnn.6

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has tensorflow1.4.0, includes: h5py, bz2file, tqdm, sympy, requests, six in anaconda python36 with cuda 8.0, and cudnn 6.0." \
      lepository="philly/jobs/custom/tensorflow" \
      tag="tf14-py36" \
      creator="v-renluo" tooltype="tensorflow" \
      tooltypeversion="1.4.0" \
      createtime="2/18/2019"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get update && apt-get install -y --no-install-recommends \
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
ENV TENSORFLOW_VERSION=1.4.0

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda create -y --name tensorflow-py$PYTHON_VERSION python=$PYTHON_VERSION numpy h5py bz2file pyyaml scipy ipython mkl tqdm six requests sympy && \
     /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/envs/tensorflow-py$PYTHON_VERSION/bin:$PATH

RUN pip install tensorflow-gpu==1.4 h5py

WORKDIR /workspace
RUN chmod -R a+w /workspace

