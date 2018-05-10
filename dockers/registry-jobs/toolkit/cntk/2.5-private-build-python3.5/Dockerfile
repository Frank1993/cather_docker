FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Labels for the docker
LABEL description="This docker has cntk 2.5 and python3.5" \
      repository="philly/jobs/custom/cntk" \
      tag="2.5.1-py35" \
      creator="vivram" \
      tooltype="cntk" \
      tooltypeversion="2.5" \
      createtime="4/25/2018"

WORKDIR /usr/local/

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libunwind-dev \
    cmake \
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    zip \
    unzip

# Allow CNTK's configure to pick up GDK and CuDNN where it expects it.
# (Note: $CUDNN_VERSION is defined by NVidia's base image)
# TODO hack, CNTK configure should be improved.
RUN _CUDNN_VERSION=$(echo $CUDNN_VERSION | cut -d. -f1-2) && \
    mkdir -p /usr/local/cudnn-$_CUDNN_VERSION/cuda/include && \
    ln -s /usr/include/cudnn.h /usr/local/cudnn-$_CUDNN_VERSION/cuda/include/cudnn.h && \
    mkdir -p /usr/local/cudnn-$_CUDNN_VERSION/cuda/lib64 && \
    ln -s /etc/alternatives/libcudnn_so /usr/local/cudnn-$_CUDNN_VERSION/cuda/lib64/libcudnn.so && \
    ln -s /usr/local/cudnn{-$_CUDNN_VERSION,} && \
    mkdir -p /usr/src/gdk/nvml/lib && \
    cp -av /usr/local/cuda/lib64/stubs/libnvidia-ml* /usr/src/gdk/nvml/lib && \
    cp -av /usr/local/cuda/lib64/stubs/libnvidia-ml.so /usr/src/gdk/nvml/lib/libnvidia-ml.so.1 && \
    mkdir -p /usr/include/nvidia/gdk && \
    cp -av /usr/local/cuda/include/nvml.h /usr/include/nvidia/gdk/nvml.h

# Install MKL for CNTK
ARG MKLML_VERSION=0.12
ARG MKLML_FILE_NAME=mklml_lnx_2018.0.1.20171227
RUN mkdir /usr/local/mklml && \
    wget --no-verbose https://github.com/01org/mkl-dnn/releases/download/v${MKLML_VERSION}/${MKLML_FILE_NAME}.tgz && \
    tar -xzf ${MKLML_FILE_NAME}.tgz -C /usr/local/mklml && \
    wget --no-verbose -O - https://github.com/01org/mkl-dnn/archive/v${MKLML_VERSION}.tar.gz | tar -xzf - && \
    cd mkl-dnn-${MKLML_VERSION} && \
    ln -s /usr/local external && \
    mkdir -p build && \
    cd build && \
    cmake .. && \
    make && \
    make install && \
    cd ../.. && \
    rm -rf mkl-dnn-${MKLML_VERSION}

# Install OpenCV
ARG OPENCV_VERSION=3.1.0
RUN wget -O /tmp/${OPENCV_VERSION}.zip https://github.com/Itseez/opencv/archive/${OPENCV_VERSION}.zip && \
    unzip -d /tmp /tmp/${OPENCV_VERSION}.zip && \
    cd /tmp/opencv-${OPENCV_VERSION} && \
    mkdir release && \
    cd release && \
    cmake -D WITH_CUDA=OFF -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local/opencv-${OPENCV_VERSION} .. && \
    make all && \
    sudo make install && \
    rm -rf /tmp/opencv-${OPENCV_VERSION}

ENV LD_LIBRARY_PATH=/usr/local/opencv-${OPENCV_VERSION}/lib:/usr/local/lib:$LD_LIBRARY_PATH

# Install python environment.
ARG MINICONDA_PREFIX=/usr/local/miniconda3
ARG MINICONDA_VERSION=4.3.11
RUN MINICONDA=Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    wget --no-verbose https://repo.continuum.io/miniconda/${MINICONDA} && \
    chmod a+x ${MINICONDA} && \
    ./${MINICONDA} -b -p ${MINICONDA_PREFIX} && \
    rm ./${MINICONDA}  && \
    ${MINICONDA_PREFIX}/bin/conda create -c defaults -c conda-forge --quiet --yes --name cntk-py35 python=3.5.2 numpy=1.11.3 scipy=0.18.1 pillow=4.0.0 librosa=0.5.0

RUN wget --no-verbose -O azcopy.tar.gz https://aka.ms/downloadazcopylinux64 && \
    mkdir -p /tmp/azcopy/ && \
    tar -xf azcopy.tar.gz -C /tmp/azcopy && \
    ls /tmp/azcopy/ && \
    sudo /tmp/azcopy/install.sh && \
    rm azcopy.tar.gz && \
    rm -rf ./tmp/azcopy/

COPY toolkit-execute /home/job/toolkit-execute
RUN chmod a+x /home/job/toolkit-execute