# Tag: nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
# Created: 2017-11-21T06:34:14.675603521Z
# Label: com.nvidia.build.id: 41212533
# Label: com.nvidia.build.ref: e0edb5359ecb7bd3d86f0c9bfa18c2260b741ebb
# Label: com.nvidia.cuda.version: 9.0.176
# Label: com.nvidia.cudnn.version: 7.0.4.31
# Ubuntu 16.04.5
FROM nvidia/cuda@sha256:33add9c50ab76b8f3a92187c0418ed600d5bea27690fda40711122fdc28ce2f4

# Labels for the docker
LABEL description="This docker has multigpu chainermn 1.3.0 and 4.1.0 versions of cupy and chainer ubuntu 16.04 version of python36 with other toolkits" \
      repository="philly/jobs/custom/chainer" \
      tag="py36_chainer_multigpu_cuda9_nobase" \
      creator="haerdoga" tooltype="chainer" \
      tooltypeversion="1.0" \
      createtime="6/11/2018"
# Everything above this line is required for the docker.

# Environment variables
ENV STAGE_DIR=/root/gpu/install \
    CUDNN_DIR=/usr/local/cudnn \
    CUDA_DIR=/usr/local/cuda-9.0 \
    OPENMPI_VERSIONBASE=1.10
ENV OPENMPI_VERSION=${OPENMPI_VERSIONBASE}.3
ENV OPENMPI_STRING=openmpi-${OPENMPI_VERSION} \
    OFED_VERSION="3.3-1.0.4.0" \
    NCCL_VERSION="2.1.15-1+cuda9.0" \
    CUPY_VERSION="4.1.0" \
    CHAINER_VERSION="4.1.0" \
    CHAINERMN_VERSION="1.3.0"

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
      sudo \
      cmake \
      g++ \
      gcc \
      net-tools \
      libsndfile-dev \
      libhdf5-dev \
      vim  && \
    apt-get autoremove

WORKDIR $STAGE_DIR

# Install Mellanox OFED user-mode drivers and its prereqs
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
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
    ./configure --with-cuda --prefix=/usr/local/${OPENMPI_STRING} && \
    make -j"$(nproc)" install && \
    rm -rf $STAGE_DIR/${OPENMPI_STRING} && \
    ln -s /usr/local/${OPENMPI_STRING} /usr/local/mpi && \
    test -f /usr/local/mpi/bin/mpic++

# Update environment variables
ENV PATH=/usr/local/mpi/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:$LD_LIBRARY_PATH

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -bfp /usr/local && \
     rm ~/miniconda.sh

RUN conda update conda && \
    conda install -y python=3 && \
    conda install -y pip && \
    pip install --upgrade pip && \
    pip install pandas h5py soundfile librosa==0.5.1 mir_eval pathos pymongo monk resampy \
                cupy==${CUPY_VERSION} chainer==${CHAINER_VERSION} chainermn==${CHAINERMN_VERSION}

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute
