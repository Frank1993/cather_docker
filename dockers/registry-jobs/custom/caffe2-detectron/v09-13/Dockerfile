FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.8.0-cudnn.6


# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

COPY detectron.zip /
RUN apt-get update && apt-get install zip unzip
RUN apt-get update && apt-get install -y --no-install-recommends expect
RUN unzip -d /detectron /detectron.zip


# Labels for the docker
LABEL description="caffe2-detectron" \
      repository="philly/jobs/custom/caffe2-detectron" \
      tag="v09-13" \
      creator="KaiHu" tooltype="caffe2-detectron" \
      tooltypeversion="1.0" \
      createtime="13/09/2018"



# caffe2 install with gpu support
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
    libopenmpi-dev \
    ninja-build \
    libprotobuf-dev \
    libsnappy-dev \
    openmpi-bin \
    openmpi-doc \
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
    flask \
    future \
    graphviz \
    hypothesis \
    jupyter \
    matplotlib \
    numpy \
    protobuf \
    pydot \
    python-nvd3 \
    pyyaml \
    requests \
    scikit-image \
    scipy \
    setuptools \
    six \
    tornado \
    tensorboardX \
    tensorflow \
    typing

RUN apt-get install -y --no-install-recommends libgflags-dev

RUN git clone --branch WCN --recursive https://github.com/YueshangGu/pytorch /pytorch 
WORKDIR /pytorch
RUN git submodule update --init --recursive

RUN cd /pytorch && mkdir build && cd build \
    && cmake .. \
    -DCUDA_ARCH_NAME=Pascal \
    -DCUDA_ARCH_BIN="35 52 60 61" \
    -DCUDA_ARCH_PTX="61" \
    -DUSE_NNPACK=OFF \
    -DUSE_ROCKSDB=OFF \
    -DUSE_NCCL=ON \
    -DNCCL_EXTERNAL=ON \
    -DUSE_GLOO=ON \
    -DUSE_MPI=ON \
    -DUSE_DISTRIBUTED=ON \
    && make -j4 install \
    && ldconfig \
    && make clean \
    && cd .. \

# ENV FULL_CAFFE2 1
# ENV FULL_CAFFE2=1
# WORKDIR /pytorch
# RUN FULL_CAFFE2=1 && python setup.py install
# WORKDIR /

# detectron install
ENV Caffe2_DIR /pytorch/build
ENV Caffe2_DIR=/pytorch/build

ENV PYTHONPATH /pytorch/build:${PYTHONPATH}
ENV PYTHONPATH=/pytorch/build:${PYTHONPATH}

ENV LD_LIBRARY_PATH /pytorch/build/lib:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/pytorch/build/lib:${LD_LIBRARY_PATH}

RUN pip install numpy>=1.13 pyyaml>=3.12 matplotlib opencv-python>=3.2 setuptools Cython mock scipy

RUN git clone https://github.com/cocodataset/cocoapi.git /cocoapi
WORKDIR /cocoapi/PythonAPI
RUN make install


WORKDIR /detectron/lib
RUN make clean
RUN make

ENV PYTHONPATH /detectron:${PYTHONPATH}
ENV PYTHONPATH=/detectron:${PYTHONPATH}

ENV LD_LIBRARY_PATH /detectron:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/detectron:${LD_LIBRARY_PATH}

# Go to Detectron root
RUN chmod -R 777 /detectron
WORKDIR /detectrons