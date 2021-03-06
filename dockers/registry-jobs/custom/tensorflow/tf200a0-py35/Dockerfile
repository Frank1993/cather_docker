FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.8.0-cudnn.6

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has tensorflow-gpu 2.0.0a0, in anaconda python35 with cuda 8.0, and cudnn 6.0." \
      repository="philly/jobs/custom/tensorflow" \
      tag="tf200a0-py35-gpu-rudder" \
      creator="joelpf" tooltype="tensorflow" \
      tooltypeversion="2.0.0a0" \
      createtime="5/8/2019"
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

ENV PYTHON_VERSION=3.5
ENV TENSORFLOW_VERSION=2.0.0a0

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda create -y --name tensorflow-py$PYTHON_VERSION python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl&& \
     /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/envs/tensorflow-py$PYTHON_VERSION/bin:$PATH

RUN pip install opencv-python && \ 
    pip install tqdm && \
    pip install tensorflow-gpu==2.0.0-alpha0 pandas h5py joblib

RUN git clone https://github.com/bioinf-jku/tensorflow-layer-library.git && \
    pip install ./tensorflow-layer-library 

RUN pip uninstall -y  Box2D-kengz && \ 
    git clone https://github.com/pybox2d/pybox2d.git && \ 
    cd pybox2d && ls && \ 
    python setup.py build && \ 
    python setup.py install && \ 
    cd .. && rm -rf pybox2d  

WORKDIR /workspace
RUN chmod -R a+w /workspace

