FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This custom base docker has pytorch 1.0.0 in anaconda python36 with CUDA 9.2, OpenMPI 3.1.1, CUDNN 7.2.1.38 " \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch1.0.0-py36-cuda9.2-openmpi3.1.1-vid" \
      creator="v-shuyg" \
      tooltype="pytorch" \
      tooltypeversion="1.0.0" \
      createtime="6/26/2019"

# Everything above this line is required for the docker.
# Add your personalized features below here.

# Environment variables
ENV STAGE_DIR=/root/gpu/install \
    CUDNN_DIR=/usr/local/cudnn \
    CUDA_DIR=/usr/local/cuda-9.2 \
    OPENMPI_VERSIONBASE=3.1 \
    NCCL_IB_DISABLE=0 \
    PYTHON_VERSION=3.6 \
    PYTORCH_VERSION=1.0.0 \
    PATH="/usr/local/mpi/bin:${PATH}" \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:$LD_LIBRARY_PATH \
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
      tmux \
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
      net-tools

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH /opt/conda/envs/pytorch-py$PYTHON_VERSION/bin:$PATH

RUN /opt/conda/bin/conda create -y --name pytorch-py$PYTHON_VERSION python=$PYTHON_VERSION && \
    /opt/conda/bin/conda clean -ya

RUN /opt/conda/bin/conda install --name pytorch-py$PYTHON_VERSION numpy scipy && \
    /opt/conda/bin/conda clean -ya

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
        libpython3.6-dev \
        libstdc++-5-dev \
        libsm6 \
        libxrender1 \
        libxext-dev \
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

# Configure OpenMPI to run good defaults:
RUN echo "hwloc_base_binding_policy = none" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "rmaps_base_mapping_policy = slot" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "btl = self,sm,tcp" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "btl_tcp_if_exclude = docker0,lo,ib0,ib1" >> /usr/local/etc/openmpi-mca-params.conf
    
# Install Pytorch
RUN /opt/conda/bin/conda install --name pytorch-py$PYTHON_VERSION torchvision cuda92 pytorch=${PYTORCH_VERSION} -c pytorch && \
    /opt/conda/bin/conda clean -ya

RUN apt-get update && apt-get install -y --no-install-recommends \
         libsndfile1 &&\
         rm -rf /var/lib/apt/lists/*

RUN /opt/conda/bin/conda install -y --name pytorch-py$PYTHON_VERSION pysoundfile tensorboardx -c conda-forge && \
    /opt/conda/bin/conda clean -ya

WORKDIR /workspace
RUN chmod -R a+w /workspace

RUN pip install --upgrade pip && \
    pip install opencv-python \
                scikit-image \
                dominate \
                dominate requests \
                dlib \
                pytz

