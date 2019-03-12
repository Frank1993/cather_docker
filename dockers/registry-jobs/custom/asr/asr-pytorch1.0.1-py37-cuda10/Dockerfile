FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

# Labels for the docker
LABEL description="This custom base docker has CUDA 10, OpenMPI 1.10.7, CUDNN 7.4.1.5, OFED 4.4-1.0.0.0, pytorch 1.0.1 python 3.7 \
             It has OpenMPI v1 and needs to be updated to OpenMPI v3" \
      repository="philly/jobs/custom/asr" \
      tag="asr-pytorch1.0.1-py37-cuda10" \
      creator="wigale" \
      tooltype="asr" \
      tooltypeversion="1.0.1" \
      createtime="3/8/2019"

# Environment variables
ENV STAGE_DIR=/root/gpu/install \
    CUDNN_DIR=/usr/local/cudnn \
    CUDA_DIR=/usr/local/cuda-10.0 \
    OPENMPI_VERSIONBASE=1.10 \
    OFED_VERSION=4.5-1.0.1.0 \
    NCCL_IB_DISABLE=1 \
    NCCL_IB_CUDA_SUPPORT=0 \
    NCCL_SOCKET_IFNAME=eth0 \
    PYTHON_VERSION=3.7 \
    PYTORCH_VERSION=1.0.1 \
    PATH=$PATH:/usr/local/mpi/bin:/usr/local/cuda-10.0/bin:/usr/local/nvidia/bin \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:/usr/local/cuda-10.0/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    LANG=C.UTF-8
ENV OPENMPI_VERSION=${OPENMPI_VERSIONBASE}.7
ENV OPENMPI_STRING=openmpi-${OPENMPI_VERSION}

RUN mkdir -p $STAGE_DIR

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
#    echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      apt-utils \
      cmake \
      git \
      curl \
      vim \
      wget \
      ca-certificates \
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
      net-tools \
    # python3.7 and pip. Replace python with python3.7
      python3.7 \
      python3.7-dev

WORKDIR $STAGE_DIR

# Install Mellanox OFED user-mode drivers and its prereqs
RUN DEBIAN_FRONTEND=noninteractive \
    # For MLNX OFED
    apt-get install -y --no-install-recommends \
        perl \
        dpkg \
        autoconf \
        libtool \
        automake1.10 \
        automake \
        m4 \
        dkms \
        debhelper \
        tcl \
        tcl8.4 \
        chrpath \
        swig \
        graphviz \
        tcl-dev \
        tcl8.4-dev \
        tk-dev \
        tk8.4-dev \
        bison \
        flex \
        dpatch \
        zlib1g-dev \
        curl \
        libcurl4-gnutls-dev \
        python-libxml2 \
        libvirt-bin \
        libvirt0 \
        libnl-3-dev \
        libglib2.0-dev \
        libgfortran3 \
        automake \
        m4 \
        pkg-config \
        libnuma-dev \
        logrotate \
        ethtool \
        lsof \
        libxslt-dev \
        libxml2-dev \
    # Philly
        dnsutils \
        pciutils \
        libltdl-dev \
        graphviz \
        tk8.4 \
        libgfortran3 \
        libnl-3-200 \
        libnl-route-3-200 \
        libnl-route-3-dev \
        libnl-utils \
        gfortran \
        tk \
        libnuma1 && \
    rm -rf /var/lib/apt/lists/* && \
    # Upgrade python
    rm -rf /usr/bin/python && \
    ln -s /usr/bin/python3.7 /usr/bin/python && \
    curl -O https://bootstrap.pypa.io/get-pip.py && \
        python get-pip.py && \
        rm get-pip.py && \
    pip install lxml && \
    apt-get autoremove && \
    wget -q -O - http://www.mellanox.com/downloads/ofed/MLNX_OFED-${OFED_VERSION}/MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu16.04-x86_64.tgz | tar xzf - && \
    cd MLNX_OFED_LINUX-$OFED_VERSION-ubuntu16.04-x86_64/DEBS && \
    for dep in libibverbs1 libibverbs-dev ibverbs-utils libmlx4-1 libmlx5-1 librdmacm1 librdmacm-dev libibumad libibumad-devel libibmad libibmad-devel; do \
        dpkg -i $dep\_*_amd64.deb; \
    done && \
    cd ../.. && \
    rm -rf MLNX_OFED_LINUX-*

##################### OPENMPI #####################

RUN apt-get update && \
    apt-get install \
        libc6-dev \
        libc6 \
        g++-5 \
        libexpat1-dev \
        libjpeg-dev \
        libpng-dev \
        libjpeg-turbo8-dev \
        libjpeg8-dev \
        libpng12-0 \
        libpython3.7-dev \
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

# Install PyTorch
RUN pip install https://download.pytorch.org/whl/cu100/torch-${PYTORCH_VERSION}-cp37-cp37m-linux_x86_64.whl



# Install LM base dependencies
RUN apt-get install \
            # Pytables dependencies
            tmux \
            vim \
            nano \
            less \
            libhdf5-dev \
            liblzo2-dev -y && \
    # Install cython and numpy before pytables so that bindings work correctly
    pip install \
        numpy \
        cython \
        python-lzo

# Install apex
RUN cd /tmp && \
    git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    python setup.py install --cuda_ext --cpp_ext && \
    rm -r /tmp/apex

# Install remaining LM dependencies
RUN    pip install \
           cffi \
           tqdm \
           spacy \
           mpi4py \
           ipdb \
           pandas \
           matplotlib \
           py3nvml \
           psutil \
           tensorboard_logger \
           tables \
           graphviz \
           git+https://github.com/szagoruyko/pytorchviz \
           git+git://github.com/pytorch/fairseq.git \
           pytorch-pretrained-bert

RUN echo "PATH=$PATH" >> /etc/environment

COPY toolkit-execute /home/job/toolkit-execute
RUN chmod a+x /home/job/toolkit-execute
WORKDIR /home/job/