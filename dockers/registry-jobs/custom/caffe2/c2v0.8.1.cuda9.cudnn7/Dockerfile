FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7
# This docker file inherit from pytorch0.3.1-py36
# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
COPY config.py /home/job/config.py
COPY train_net.py /home/job/train_net.py
RUN chmod a+rw /home/job/config.py /home/job/train_net.py \
    && chmod u+x /home/job/toolkit-execute

# Labels for the docker 
LABEL description="This docker has caffe2 0.8.1 in python27 with cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/caffe2" \
      tag="c2v0.8.1.cuda9.cudnn7" \
      creator="ruixia" tooltype="caffe2" \
      tooltypeversion="0.8.1" \
      createtime="4/20/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

# Migrated from: https://github.com/pytorch/pytorch/blob/master/docker/caffe2/ubuntu-16.04-cuda8-cudnn7-all-options/Dockerfile

# caffe2 install with gpu support
RUN rm -rf /etc/apt/sources.list.d/*
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgflags-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libiomp-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libprotobuf-dev \
    libsnappy-dev \
    protobuf-compiler \
    python-dev \
    python-numpy \
    python-pip \
    python-pydot \
    python-setuptools \
    python-scipy \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir \
    Cython \
    flask \
    future \
    graphviz \
    hypothesis \
    jupyter \
    matplotlib \
    mock \
    numpy>=1.13 \
    opencv-python>=3.2 \
    protobuf \
    pydot \
    python-nvd3 \
    pyyaml>=3.12 \
    requests \
    scikit-image \
    scipy \
    setuptools \
    six \
    tornado

########## INSTALLATION STEPS ###################
RUN git clone https://github.com/caffe2/caffe2.git

# The following command is a workaround to get the correct aten instead returning error
RUN cd caffe2/third_party \
    && git clone https://github.com/zdevito/ATen aten \
    && cd aten \
    && git checkout 642baf51c5b7e13ad814542c0b47ab03a14f8c92 \
    && cd .. \
    && git add aten \
    && git clone https://github.com/eigenteam/eigen-git-mirror eigen \
    && cd eigen \
    && git checkout 8b2c28cc4e52c62cbfd457fd94bdf1e91b4f9603 \
    && cd .. \
    && git add eigen

RUN cd caffe2 && git submodule update --init --recursive \
    && mkdir build && cd build \
    && cmake .. \
    -DCUDA_ARCH_NAME=Manual \
    -DCUDA_ARCH_BIN="35 52 60 61" \
    -DCUDA_ARCH_PTX="61" \
    -DUSE_NNPACK=OFF \
    -DUSE_NCCL=ON \
    -DUSE_ROCKSDB=OFF \
    -DCMAKE_INSTALL_PREFIX=/usr/local/caffe2 \
    && make -j"$(nproc)" install \
    && ldconfig \
    && make clean \
    && cd .. \
    && rm -rf build

RUN git clone https://github.com/cocodataset/cocoapi.git \
    && cd cocoapi/PythonAPI \
    && make install

RUN git clone https://github.com/harrysummer/Detectron.git /detectron \
    && cd /detectron \
    && make \
    && chmod -R a+w /detectron

ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/caffe2/lib

# Use different method to ensure PYTHONPATH is set properly
ENV PYTHONPATH $PYTHONPATH:/usr/local/caffe2/lib/python2.7/dist-packages:/detectron
ENV PYTHONPATH=$PYTHONPATH:/usr/local/caffe2/lib/python2.7/dist-packages:/detectron
RUN echo "export PYTHONPATH=/usr/local/caffe2/lib/python2.7/dist-packages:/detectron" > /etc/profile.d/pythonpath.sh
RUN mkdir ~/.ssh && echo "export PYTHONPATH=/usr/local/caffe2/lib/python2.7/dist-packages:/detectron" >> ~/.ssh/environment

WORKDIR /workspace
RUN chmod -R a+w /workspace
