FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 1.0rc1 in anaconda python36 with cuda 10.0, cuDNN 7.3.1.20, OpenMPI 1.10.3, NCCL 2.3.5-2." \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch1.0rc1-py36-cuda10.0-openmpi1.10.3-face" \
      creator="v-ximing" tooltype="pytorch" \
      tooltypeversion="1.0rc1" \
      createtime="11/04/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

# Environment variables
ENV STAGE_DIR=/root/gpu/install \
    CUDNN_DIR=/usr/local/cudnn \
    CUDA_DIR=/usr/local/cuda-10.0 \
    CUDNN_VERSION=${CUDNN_VERSION}+cuda10.0 \
    NCCL_VERSION=2.3.5-2+cuda10.0 \
    NCCL_IB_DISABLE=0 \
    PYTHON_VERSION=3.6 \
    PYTORCH_VERSION=1.0rc1 \
    LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH} \
    LANG=C.UTF-8 \
    OPENMPI_VERSIONBASE=1.10
ENV OPENMPI_VERSION=${OPENMPI_VERSIONBASE}.3
ENV OPENMPI_STRING=openmpi-${OPENMPI_VERSION} \
    OFED_VERSION=4.2-1.2.0.0

RUN mkdir -p $STAGE_DIR

RUN apt-get update && \
    apt-get install -y \
      apt-utils \
      software-properties-common \
      autotools-dev \
      htop \
      tree \
      vim \
      build-essential \
      cmake \
      g++-5 \
      gcc-5 \
      git \
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
    # ifconfig
      net-tools \
      ca-certificates \
      libjpeg-dev \
      libpng-dev \
      ninja-build && \
    apt-get autoremove && \
    apt-get autoclean

WORKDIR $STAGE_DIR

# Install Mellanox OFED user-mode drivers and its prereqs
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
    # For MLNX OFED
        dnsutils \
        pciutils \
        ethtool \
        lsof \
        python-libxml2 \
        quilt \
        libltdl-dev \
        dpatch \
        autotools-dev \
        graphviz \
        autoconf \
        chrpath \
        swig \
        automake \
        tk8.4 \
        tcl8.4 \
        libgfortran3 \
        tcl \
        libnl-3-200 \
        libnl-route-3-200 \
        libnl-route-3-dev \
        libnl-utils \
        gfortran \
        tk \
        bison \
        flex \
        libnuma1 && \
    rm -rf /var/lib/apt/lists/* && \
    wget -q -O - http://www.mellanox.com/downloads/ofed/MLNX_OFED-$OFED_VERSION/MLNX_OFED_LINUX-$OFED_VERSION-ubuntu16.04-x86_64.tgz | tar xzf - && \
    cd MLNX_OFED_LINUX-$OFED_VERSION-ubuntu16.04-x86_64/DEBS && \
    for dep in libibverbs1 libibverbs-dev ibverbs-utils libmlx4-1 libmlx5-1 librdmacm1 librdmacm-dev libibumad libibumad-devel libibmad libibmad-devel; do \
        dpkg -i $dep\_*_amd64.deb; \
    done && \
    cd ../.. && \
    rm -rf MLNX_OFED_LINUX-*

##################### OPENMPI #####################

RUN wget -q -O - https://www.open-mpi.org/software/ompi/v${OPENMPI_VERSIONBASE}/downloads/${OPENMPI_STRING}.tar.gz | tar -xzf - && \
    cd ${OPENMPI_STRING} && \
    ./configure --prefix=/usr/local/${OPENMPI_STRING} && \
    make -j"$(nproc)" install && \
    rm -rf $STAGE_DIR/${OPENMPI_STRING} && \
    ln -s /usr/local/${OPENMPI_STRING} /usr/local/mpi && \
    # Sanity check:
    test -f /usr/local/mpi/bin/mpic++

# Update environment variables
ENV PATH=/usr/local/mpi/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:$LD_LIBRARY_PATH

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda create -y --name pytorch-py${PYTHON_VERSION} python=${PYTHON_VERSION} \
        numpy pyyaml scipy ipython mkl matplotlib mkl-include setuptools cmake cffi typing && \
     /opt/conda/bin/conda install --name pytorch-py${PYTHON_VERSION} -c mingfeima mkldnn && \
     /opt/conda/bin/conda clean -ya

ENV PATH=/opt/conda/envs/pytorch-py$PYTHON_VERSION/bin:${PATH}
ENV CMAKE_PREFIX_PATH=/opt/conda/envs/pytorch-py${PYTHON_VERSION}

RUN git clone --recursive -b v1.0rc1 --single-branch https://github.com/pytorch/pytorch ~/pytorch
RUN cd ~/pytorch && python setup.py install

RUN git clone https://github.com/pytorch/vision.git ~/torchvision
RUN cd ~/torchvision && python setup.py install

RUN rm -r ~/pytorch ~/torchvision

WORKDIR /workspace
RUN chmod -R a+w /workspace
