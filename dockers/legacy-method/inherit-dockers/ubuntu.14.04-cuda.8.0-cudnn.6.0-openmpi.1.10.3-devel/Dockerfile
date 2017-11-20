# Tag: nvidia/cuda:8.0-cudnn6-devel-ubuntu14.04
# Created: 2017-05-24T19:12:39.834912407Z
# Label: com.nvidia.build.id: 17095823
# Label: com.nvidia.build.ref: 77e8a390c5e26257b3331bf9f208b3096dd6dbdf
# Label: com.nvidia.cuda.version: 8.0.61
# Label: com.nvidia.cudnn.version: 6.0.21
# Ubuntu 14.04.5
FROM nvidia/cuda@sha256:eba04e3cae5e8a2e83fdcc121ab5e1a15c98d0a8b895539aff5fc76d18b06375

# Environment variables
ENV STAGE_DIR=/root/gpu/install \
    CUDNN_DIR=/usr/local/cudnn \
    CUDA_DIR=/usr/local/cuda-8.0 \
    OPENMPI_VERSIONBASE=1.10
ENV OPENMPI_VERSION=${OPENMPI_VERSIONBASE}.3
ENV OPENMPI_STRING=openmpi-${OPENMPI_VERSION}

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
    # Needed by OpenMPI
      cmake \
      g++ \
      gcc && \
    apt-get autoremove

WORKDIR $STAGE_DIR

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
ENV PATH=/usr/local/mpi/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:$LD_LIBRARY_PATH
