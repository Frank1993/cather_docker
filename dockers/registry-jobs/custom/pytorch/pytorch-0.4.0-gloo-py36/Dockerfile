FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# This docker file inherit from base philly-openmpi and install cuda-aware mpi & pytorch & torchvision & nltk & gym
# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 0.4.0 torchvision in anaconda python36 with cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch-0.4.0-gloo-py36" \
      creator="v-shuzhe" tooltype="pytorch" \
      tooltypeversion="0.4.0" \
      createtime="4/6/2018"

RUN apt-get update && apt-get install -y --no-install-recommends \
         git \
         vim \
         ca-certificates \
         libnccl2 \
         libnccl-dev \
         libjpeg-dev \
         libpng-dev &&\
         rm -rf /var/lib/apt/lists/*

ENV PYTHON_VERSION=3.6

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH 
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN cd ~ &&\
    conda install -y numpy pyyaml scipy ipython mkl mkl-include setuptools cmake cffi typing && \
    conda install -y cuda91 -c pytorch&& \
    conda install -y -c mingfeima mkldnn &&\
    conda install -y -c pytorch magma-cuda91 &&\
    conda install -y cffi tqdm &&\
    conda install -y opencv matplotlib && \
# conda search openmpi -c conda-forge -----> only have 3.1.0 version openmpi, seems this version can not pass philly's hostname validation .
#    conda install -y -c conda-forge openmpi &&\
    conda install -y -c pytorch pytorch=0.4.0 torchvision nltk && \
    conda clean -ya 


