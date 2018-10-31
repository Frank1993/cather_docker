FROM nvidia/cuda@sha256:33add9c50ab76b8f3a92187c0418ed600d5bea27690fda40711122fdc28ce2f4

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod 777 /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has tensorflow latest using python36, with packages required by tensorflow object detection api" \
      repository="philly/jobs/custom/tf_fastrcnn" \
      tag="tf110-py36" \
      creator="v-minghl" tooltype="tf_fastrcnn" \
      tooltypeversion="1.0" \
      createtime="9/14/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.
ENV CUDNN_DIR=/usr/local/cudnn \
    CUDNN_VERSION=7.2.1.38-1+cuda9.0 


# Environment variables
ENV STAGE_DIR=/root/gpu/install \
    CUDNN_DIR=/usr/local/cudnn \
    CUDNN_VERSION=7.2.1.38-1+cuda9.0 \
    CUDA_DIR=/usr/local/cuda-9.0 \
    OPENMPI_VERSIONBASE=1.10
ENV OPENMPI_VERSION=${OPENMPI_VERSIONBASE}.3
ENV OPENMPI_STRING=openmpi-${OPENMPI_VERSION} \
    OFED_VERSION=4.2-1.2.0.0

RUN mkdir -p $STAGE_DIR

RUN apt-get -y update && \
    apt-get -y install \
      build-essential \
      autotools-dev \
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
      net-tools && \
    apt-get autoremove

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
        libnuma1 \
        checkinstall && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/* && \
    # libnl1 is not available in ubuntu16 so build from source
    wget -q -O - http://www.infradead.org/~tgr/libnl/files/libnl-1.1.4.tar.gz | tar xzf - && \
    cd libnl-1.1.4 && \
    ./configure && \
    make && \
    checkinstall -D --showinstall=no --install=yes -y -pkgname=libnl1 -A amd64 && \
    cd .. && \
    rm -rf libnl-1.1.4 && \
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



# Environment variables
ENV STAGE_DIR=/root/gpu/install \
    CUDNN_DIR=/usr/local/cudnn \
    CUDNN_VERSION=7.3.0.29-1+cuda9.0 \
    CUDA_DIR=/usr/local/cuda-9.0 \
    OPENMPI_VERSIONBASE=1.10
ENV OPENMPI_VERSION=${OPENMPI_VERSIONBASE}.3
ENV OPENMPI_STRING=openmpi-${OPENMPI_VERSION} \
    OFED_VERSION=4.2-1.2.0.0

RUN mkdir -p $STAGE_DIR

RUN apt-get -y update && \
    apt-get -y install \
      build-essential \
      autotools-dev \
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
      net-tools && \
    apt-get autoremove

WORKDIR $STAGE_DIR

RUN apt-get update && apt-get install -y --no-install-recommends --force-yes \
            libcudnn7=$CUDNN_VERSION \
            libcudnn7-dev=$CUDNN_VERSION && \
    rm -rf /var/lib/apt/lists/*

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
        libnuma1 \
        checkinstall && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/* && \
    # libnl1 is not available in ubuntu16 so build from source
    wget -q -O - http://www.infradead.org/~tgr/libnl/files/libnl-1.1.4.tar.gz | tar xzf - && \
    cd libnl-1.1.4 && \
    ./configure && \
    make && \
    checkinstall -D --showinstall=no --install=yes -y -pkgname=libnl1 -A amd64 && \
    cd .. && \
    rm -rf libnl-1.1.4 && \
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


RUN apt-get update
RUN apt-get -y install software-properties-common libcupti-dev

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get -y update && \
    apt-get -y install python3.5 python3.5-dev python3.5-tk 
RUN curl https://bootstrap.pypa.io/get-pip.py | sudo python3.5

RUN python3.5 -m pip install --upgrade numpy scipy h5py cython easydict ipython opencv-python setproctitle msgpack msgpack_numpy zmq tqdm matplotlib contextlib2 pillow lxml jupyter
RUN python3.5 -m pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.11.0-cp35-cp35m-linux_x86_64.whl

ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH


