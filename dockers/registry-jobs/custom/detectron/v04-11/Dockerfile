FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.8.0-cudnn.6


# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

COPY detectron.zip /
RUN apt-get update && apt-get install zip unzip
RUN unzip -d /detectron /detectron.zip


# Labels for the docker
LABEL description="caffe2-detectron" \
      repository="philly/jobs/custom/detectron" \
      tag="v03-23" \
      creator="seliang" tooltype="caffe2-detectron" \
      tooltypeversion="1.0" \
      createtime="23/03/2018"



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

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
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
    tornado

RUN git clone --branch master --recursive https://github.com/caffe2/caffe2.git /caffe2

RUN cd /caffe2 && mkdir build && cd build \
    && cmake .. \
    -DCUDA_ARCH_NAME=Pascal \
    -DCUDA_ARCH_BIN="35 52 60 61" \
    -DCUDA_ARCH_PTX="61" \
    -DUSE_NNPACK=OFF \
    -DUSE_ROCKSDB=OFF \
    && make -j4 install \
    && ldconfig \
    && make clean \
    && cd .. \


# detectron install
ENV Caffe2_DIR /caffe2/build
ENV Caffe2_DIR=/caffe2/build

ENV PYTHONPATH /caffe2/build:${PYTHONPATH}
ENV PYTHONPATH=/caffe2/build:${PYTHONPATH}

ENV LD_LIBRARY_PATH /caffe2/build/lib:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/caffe2/build/lib:${LD_LIBRARY_PATH}

RUN pip install numpy>=1.13 pyyaml>=3.12 matplotlib opencv-python>=3.2 setuptools Cython mock scipy

RUN git clone https://github.com/cocodataset/cocoapi.git /cocoapi
WORKDIR /cocoapi/PythonAPI
RUN make install


WORKDIR /detectron/lib
RUN make clean
RUN make

ENV PYTHONPATH /detectron/lib:${PYTHONPATH}
ENV PYTHONPATH=/detectron/lib:${PYTHONPATH}

ENV LD_LIBRARY_PATH /detectron/lib:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/detectron/lib:${LD_LIBRARY_PATH}

# Go to Detectron root
RUN chmod -R 777 /detectron
WORKDIR /detectron