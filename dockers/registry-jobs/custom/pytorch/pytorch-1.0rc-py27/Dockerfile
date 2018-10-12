FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# This docker file inherit from base philly-openmpi and install cuda-aware mpi & pytorch & torchvision & nltk & gym
# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 1.0rc torchvision in anaconda python27 with cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch-1.0rc-py27" \
      creator="quyou" tooltype="pytorch" \
      tooltypeversion="1.0rc" \
      createtime="10/12/2018"

COPY nccl_2.3.5_16.04_cuda8.deb /home/job
RUN dpkg -i /home/job/nccl_2.3.5_16.04_cuda8.deb

RUN apt-get update && apt-get install -y --no-install-recommends \
         git \
         vim \
         ca-certificates \
         libnccl2=2.3.5-2+cuda9.0 \
         libnccl-dev=2.3.5-2+cuda9.0 \
         libjpeg-dev \
         libpng-dev \
         unzip &&\
         rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH 
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN conda install -y python=2.7 numpy pyyaml scipy ipython mkl scikit-learn matplotlib pandas setuptools Cython h5py graphviz 

RUN conda install mkl-include cmake cffi typing
RUN conda install -c mingfeima mkldnn
RUN conda clean -ya

COPY requirements.txt /home/job

RUN /opt/conda/bin/pip install -r /home/job/requirements.txt && \
#    git clone --recursive https://github.com/pytorch/pytorch && cd pytorch && python setup.py install &&\
    conda install -c pytorch pytorch-nightly  && \
    cd ~ && git clone --recursive https://github.com/pytorch/vision.git && cd vision && python setup.py install 

RUN export CUDA_HOME="/usr/local/cuda"



