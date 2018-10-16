FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 0.4.1 in anaconda python36 with cuda 8.0, OpenMPI 3.1.1, cuDNN 7.2.1.38, NCCL 2.3.5-2." \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch0.4.1-py36-cuda8.0-openmpi3.1.1-face" \
      creator="v-ximing" tooltype="pytorch" \
      tooltypeversion="0.4.1" \
      createtime="10/16/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

# Environment variables
ENV STAGE_DIR=/root/gpu/install \
    CUDNN_DIR=/usr/local/cudnn \
    CUDA_DIR=/usr/local/cuda-8.0 \
    CUDNN_VERSION=${CUDNN_VERSION}+cuda8.0 \
    OPENMPI_VERSIONBASE=3.1 \
    NCCL_VERSION=2.3.5-2+cuda8.0 \
    NCCL_IB_DISABLE=0 \
    PYTHON_VERSION=3.6 \
    PYTORCH_VERSION=0.4.1 \
    PATH=/usr/local/mpi/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:${LD_LIBRARY_PATH} \
    LANG=C.UTF-8
ENV OPENMPI_VERSION=${OPENMPI_VERSIONBASE}.1
ENV OPENMPI_STRING=openmpi-${OPENMPI_VERSION}

RUN mkdir -p $STAGE_DIR

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      git \
      curl \
      vim \
      wget \
      ca-certificates \
      libnccl2=${NCCL_VERSION} \
      libnccl-dev=${NCCL_VERSION} \
      rsync \
      curl \
      wget \
      jq \
      openssh-server \
      openssh-client \
    # No longer in 'minimal set of packages'
      sudo \
    # Needed by OpenMPI
      cmake \
      g++ \
      gcc \
    # ifconfig
      net-tools

##################### OPENMPI #####################
RUN apt-get update && \
    apt-get install \
        libc6-dev \
        libc6 \
        g++-5 \
        gcc-5 \
        libexpat1-dev \
        libjpeg-dev \
        libpng-dev \
        libjpeg-turbo8-dev \
        libjpeg8-dev \
        libpng12-0 \
        libpython3.6-dev \
        libstdc++-5-dev \
        zlib1g-dev -y && \
    ln -s /usr/include/asm-generic/ /usr/include/asm && \
    mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v${OPENMPI_VERSIONBASE}/downloads/${OPENMPI_STRING}.tar.gz && \
    tar zxf ${OPENMPI_STRING}.tar.gz && \
    cd ${OPENMPI_STRING} && \
    ./configure --enable-orterun-prefix-by-default --with-openib --prefix /usr/local/mpi  && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi && \
    test -f /usr/local/mpi/bin/mpic++     # Sanity check

# Configure NCCL and OpenMPI to run good defaults:
RUN echo "hwloc_base_binding_policy = none" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "rmaps_base_mapping_policy = slot" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "btl = self,sm,tcp" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "btl_tcp_if_exclude = docker0,lo,ib0,ib1" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo NCCL_DEBUG=INFO >> /etc/nccl.conf && \
    echo NCCL_SOCKET_IFNAME=^docker0 >> /etc/nccl.conf

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    libjpeg-dev \
    curl \
    cmake \
    ninja-build \
    libpng-dev

RUN rm -rf /var/lib/apt/lists/*
RUN apt-get autoclean

ENV PYTHON_VERSION=3.6

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda create -y --name pytorch-py${PYTHON_VERSION} python=${PYTHON_VERSION} \
        numpy pyyaml scipy ipython mkl matplotlib mkl-include setuptools cmake cffi typing && \
     /opt/conda/bin/conda install --name pytorch-py${PYTHON_VERSION} -c mingfeima mkldnn && \
     /opt/conda/bin/conda install --name pytorch-py${PYTHON_VERSION} -c pytorch magma-cuda80 && \
     /opt/conda/bin/conda clean -ya

ENV PATH=/opt/conda/envs/pytorch-py$PYTHON_VERSION/bin:${PATH}
ENV CMAKE_PREFIX_PATH=/opt/conda/envs/pytorch-py${PYTHON_VERSION}
ENV TORCH_CUDA_ARCH_LIST="3.5;3.7;5.2;6.0;6.1"

RUN git clone --recursive https://github.com/pytorch/pytorch ~/pytorch
RUN cd ~/pytorch && git checkout v0.4.1 && git submodule update --init && python setup.py install

RUN git clone https://github.com/pytorch/vision.git ~/torchvision
RUN cd ~/torchvision && python setup.py install

RUN rm -r ~/pytorch ~/torchvision

WORKDIR /workspace
RUN chmod -R a+w /workspace
