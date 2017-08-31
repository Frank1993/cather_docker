# Tag: nvidia/cuda:8.0-cudnn5-devel-ubuntu14.04
# Created: 2017-08-11T17:04:39.727532619Z
# Label: com.nvidia.build.id: 29071385
# Label: com.nvidia.build.ref: a6594cbeaf2df0f10125cf173af2137c3fda61b2
# Label: com.nvidia.cuda.version: 8.0.61
# Label: com.nvidia.cudnn.version: 5.1.10
# Ubuntu 14.04.5
FROM nvidia/cuda@sha256:a84fb6b231a9c8ac74942ef6bfd67886b50a4918ef40aa1eae97688ff09fc56d

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
      openjdk-7-jre \
      openjdk-7-jdk \
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
ENV PATH=/usr/local/mpi/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:$LD_LIBRARY_PATH \
    JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd6
