FROM nvidia/cuda:8.0-cudnn5-devel

# Environment variables
ENV CUB_VERSIONS="1.4.1 1.5.2" \
    STAGE_DIR=/root/gpu/install \
    CUDNN_DIR=/usr/local/cudnn \
    CUDA_DIR=/usr/local/cuda-8.0 \
    OPENMPI_VERSIONBASE=1.10
ENV OPENMPI_VERSION=${OPENMPI_VERSIONBASE}.3
ENV OPENMPI_STRING=openmpi-${OPENMPI_VERSION}

RUN mkdir -p $STAGE_DIR

RUN apt-get -y update && \
    apt-get -y install \ 
      build-essential \
      wget \
      autotools-dev \
      byacc \
      unzip \
      cmake \
      pkg-config \
      gdb \
      gcc \
      nano \
      vim \
      joe \
      curl \
      jq \
      gawk \
      psmisc \
      openssh-server \
      openssh-client \
      git \
      inotify-tools \
      rsync \
      realpath \
   # Used in MVAPICH2 MPI install:
      rpm2cpio

WORKDIR $STAGE_DIR

##################### NVIDIA CUB #####################

# Not creating link as CNTK uses particular version of CUB and needs to be tested with every new one.
RUN for CUB_VERSION in $CUB_VERSIONS; do \
      CUB_STRING=cub-$CUB_VERSION; \
      wget -O - --no-verbose https://github.com/NVlabs/cub/archive/$CUB_VERSION.tar.gz | tar -xzf - -C /usr/local; \
      test -f /usr/local/$CUB_STRING/cub/cub.cuh; \
    done

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
