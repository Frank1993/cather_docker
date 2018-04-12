FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has multigpu chainermn and 4.0.0rc1 versions of cupy and chainer ubuntu 16.04 version of python36 with other toolkits" \
      repository="philly/jobs/custom/chainer" \
      tag="py36_chainer_multigpu_cuda9" \
      creator="haerdoga" tooltype="chainer" \
      tooltypeversion="1.0" \
      createtime="4/11/2018"
# Everything above this line is required for the docker.

ENV NCCL_VERSION="2.1.15-1+cuda9.0" \
    CUPY_VERSION="4.0.0rc1" \
    CHAINER_VERSION="4.0.0rc1" \
    CHAINERMN_VERSION="1.2.0"

##################### OPENMPI #####################

# reinstall openMPI with --with-cuda option for chainermn
# first install hwloc version 1.11.10 separately
RUN cd $STAGE_DIR && wget -q -O - https://www.open-mpi.org/software/hwloc/v1.11/downloads/hwloc-1.11.10.tar.gz | tar -xzf - && \
    cd hwloc-1.11.10 && \
    ./configure --prefix=/usr/local/hwloc-1.11.10 && \
    make -j"$(nproc)" install && \
    rm -rf $STAGE_DIR/hwloc-1.11.10 && \
    cd $STAGE_DIR && \
    wget -q -O - https://www.open-mpi.org/software/ompi/v${OPENMPI_VERSIONBASE}/downloads/${OPENMPI_STRING}.tar.gz | tar -xzf - && \
    cd ${OPENMPI_STRING} && \
    ./configure --with-cuda --with-hwloc=/usr/local/hwloc-1.11.10 --prefix=/usr/local/${OPENMPI_STRING} && \
    make -j"$(nproc)" install && \
    rm -rf $STAGE_DIR/${OPENMPI_STRING} && \
    ln -s /usr/local/${OPENMPI_STRING} /usr/local/mpi && \
    test -f /usr/local/mpi/bin/mpic++

# this repo contains nvidia stuff
RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

# install libraries from the repos
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libnccl2=${NCCL_VERSION} \
    libnccl-dev=${NCCL_VERSION} \
    libsndfile-dev \
    libhdf5-dev \
    vim  && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -bfp /usr/local && \
     rm ~/miniconda.sh

RUN conda update conda && \
    conda install -y python=3 && \
    conda install -y pip && \
    pip install --upgrade pip && \
    pip install pandas h5py soundfile librosa mir_eval pathos pymongo monk resampy \
                cupy==${CUPY_VERSION} chainer==${CHAINER_VERSION} chainermn==${CHAINERMN_VERSION}

