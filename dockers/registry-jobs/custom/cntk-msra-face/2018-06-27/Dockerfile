FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.14.04-cuda.8.0-cudnn.6

# Tag: nvidia/cuda:8.0-cudnn6-devel-ubuntu14.04
# Created: 2017-05-24T19:12:39.834912407Z
# Label: com.nvidia.build.id: 17095823
# Label: com.nvidia.build.ref: 77e8a390c5e26257b3331bf9f208b3096dd6dbdf
# Label: com.nvidia.cuda.version: 8.0.61
# Label: com.nvidia.cudnn.version: 6.0.21
# Ubuntu 14.04.5
#FROM nvidia/cuda@sha256:eba04e3cae5e8a2e83fdcc121ab5e1a15c98d0a8b895539aff5fc76d18b06375

# Labels for the docker

RUN mkdir /home/job
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

LABEL description="custom cntk for face team" \
      repository="philly/jobs/custom/cntk-msra-face" \
      tag="2018-06-27" \
      creator="lewlu" \
      tooltype="cntk"

RUN DEBIAN_FRONTEND=noninteractive && \
    apt-get update && apt-get install -y --no-install-recommends \
        autotools-dev \
        build-essential \
        cmake \
        g++ \
        git ca-certificates \
        libavcodec-dev \
        libavformat-dev \
        libjasper-dev \
        libjpeg-dev \
        libpng-dev \
        liblapacke-dev \
        libswscale-dev \
        libtiff-dev \
        pkg-config \
        wget \
        zlib1g-dev \
        # Protobuf
        ca-certificates \
        curl \
        unzip \
        automake \
        libtool \
        autoconf \
        # Boost
        libbz2-dev \
        # Java Bindings
        openjdk-7-jdk \
        # Test dependencies
        gawk \
        python \
        python-yaml \
        python-six \
        rsync \
        # SWIG
        libpcre++-dev \
        # OpenBLAS
        gfortran \
        # Kaldi
        python-dev \
        automake \
        libtool \
        autoconf \
        subversion \
        # Kaldi's dependencies
        libapr1 libaprutil1 libltdl-dev libltdl7 libserf-1-1 libsigsegv2 libsvn1 m4 \
        # Additional dependencies
        rsync \
        curl \
        wget \
        jq \
        openssh-server \
        openssh-client \
        && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root

ARG PROTOBUF_VERSION=3.1.0
RUN PROTOBUF_STRING=protobuf-$PROTOBUF_VERSION && \
    wget -O - --no-verbose https://github.com/google/protobuf/archive/v${PROTOBUF_VERSION}.tar.gz | tar -xzf - && \
    cd $PROTOBUF_STRING && \
    ./autogen.sh && \
    ./configure CFLAGS=-fPIC CXXFLAGS=-fPIC --disable-shared --prefix=/usr/local/$PROTOBUF_STRING && \
    make -j $(nproc) install && \
    cd .. && \
    rm -rf $PROTOBUF_STRING

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

ARG CUB_VERSION=1.4.1
RUN wget -q -O - https://github.com/NVlabs/cub/archive/$CUB_VERSION.tar.gz | tar -C /usr/local -xzf -

ARG SWIG_VERSION=3.0.10
RUN wget -q http://prdownloads.sourceforge.net/swig/swig-$SWIG_VERSION.tar.gz -O - | tar xvfz - && \
    cd swig-$SWIG_VERSION && \
    ./configure --prefix=/usr/local/swig-$SWIG_VERSION --without-perl5 && \
    make -j $(nproc) && \
    make install && \
    cd .. && \
    rm -rf swig-$SWIG_VERSION

ARG BOOST_VERSION=1.60.0
RUN BOOST_UNDERSCORE_VERSION=$(echo $BOOST_VERSION | tr . _) && \
    wget -q -O - https://sourceforge.net/projects/boost/files/boost/${BOOST_VERSION}/boost_${BOOST_UNDERSCORE_VERSION}.tar.gz/download | tar -xzf - && \
    cd boost_${BOOST_UNDERSCORE_VERSION} && \
    ./bootstrap.sh --prefix=/usr/local/boost-${BOOST_VERSION}  && \
    ./b2 -d0 -j $(nproc) install && \
    cd .. && \
    rm -rf boost_${BOOST_UNDERSCORE_VERSION}

ARG OPENBLAS_VERSION=0.2.18
RUN wget -q -O - https://github.com/xianyi/OpenBLAS/archive/v${OPENBLAS_VERSION}.tar.gz | tar -xzf - && \
    cd OpenBLAS-${OPENBLAS_VERSION} && \
    make -j 2 MAKE_NB_JOBS=0 USE_OPENMP=1 | tee make.log && \
    grep -qF 'OpenBLAS build complete. (BLAS CBLAS LAPACK LAPACKE)' make.log && \
    grep -qF 'Use OpenMP in the multithreading.' make.log && \
    make PREFIX=/usr/local/openblas install && \
    cd .. && \
    rm -rf OpenBLAS-${OPENBLAS_VERSION}

ENV LD_LIBRARY_PATH /usr/local/openblas/lib:$LD_LIBRARY_PATH

ARG LIBZIP_VERSION=1.1.2
RUN wget -q -O - http://nih.at/libzip/libzip-${LIBZIP_VERSION}.tar.gz | tar -xzf - && \
    cd libzip-${LIBZIP_VERSION} && \
    ./configure && \
    make -j $(nproc) install && \
    cd .. && \
    rm -rf libzip-${LIBZIP_VERSION}
ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH

ARG OPENCV_VERSION=3.1.0
RUN wget -q -O - https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.tar.gz | tar -xzf - && \
    cd opencv-${OPENCV_VERSION} && \
    cmake -DWITH_CUDA=OFF -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local/opencv-${OPENCV_VERSION} . && \
    make -j $(nproc) install && \
    cd .. && \
    rm -rf opencv-${OPENCV_VERSION}

ARG OFED_VERSION=3.3-1.0.4.0
RUN DEBIAN_FRONTEND=noninteractive && \
    apt-get -y update && \
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
        libnl1 \
        gfortran \
        tk \
        bison \
        flex \
        libnuma1 \
        && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/* && \
    wget -q -O - http://www.mellanox.com/downloads/ofed/MLNX_OFED-$OFED_VERSION/MLNX_OFED_LINUX-$OFED_VERSION-ubuntu14.04-x86_64.tgz | tar xzf - && \
    cd MLNX_OFED_LINUX-$OFED_VERSION-ubuntu14.04-x86_64/DEBS && \
    for dep in libibverbs1 libibverbs-dev ibverbs-utils libmlx4-1 libmlx5-1 librdmacm1 librdmacm-dev libibumad libibumad-devel libibmad libibmad-devel; do \
        dpkg -i $dep\_*_amd64.deb; \
    done && \
    cd ../.. && \
    rm -rf MLNX_OFED_LINUX-*

ARG OPENMPI_VERSION=1.10.7
RUN wget -q -O - https://www.open-mpi.org/software/ompi/v1.10/downloads/openmpi-${OPENMPI_VERSION}.tar.gz | tar -xzf - && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --with-verbs --with-cuda=/usr/local/cuda --prefix=/usr/local/mpi && \
    make -j $(nproc) install && \
    cd .. && \
    rm -rf openmpi-${OPENMPI_VERSION}
ENV PATH /usr/local/mpi/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/mpi/lib:$LD_LIBRARY_PATH

ARG CNTK_CUSTOM_MKL_VERSION=3
RUN mkdir /usr/local/CNTKCustomMKL && \
    wget --no-verbose -O - https://www.cntk.ai/mkl/CNTKCustomMKL-Linux-$CNTK_CUSTOM_MKL_VERSION.tgz | \
    tar -xzf - -C /usr/local/CNTKCustomMKL

ARG MKLML_VERSION=v0.11/mklml_lnx_2018.0.1.20171007
RUN mkdir /usr/local/mklml && \
    wget --no-verbose -O - https://github.com/01org/mkl-dnn/releases/download/$MKLML_VERSION.tgz | \
    tar -xzf - -C /usr/local/mklml

ARG NCCL_VERSION=2.0.5-2
RUN wget --no-verbose http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64/libnccl2_${NCCL_VERSION}+cuda8.0_amd64.deb && \
    wget --no-verbose http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64/libnccl-dev_${NCCL_VERSION}+cuda8.0_amd64.deb && \
    dpkg -i libnccl2_${NCCL_VERSION}+cuda8.0_amd64.deb && \
    dpkg -i libnccl-dev_${NCCL_VERSION}+cuda8.0_amd64.deb && \
    rm libnccl*


RUN chmod -R 777 /home/

RUN cd /home/ && \
    git clone -b lewlu/AM-Softmax https://github.com/Microsoft/CNTK.git && \
    mkdir CNTK/build/release -p && \
    cd CNTK/build/release && \
    ../../configure --asgd=no && \
    make clean && make -j 2 && \
    cd bin && \
    chmod 777 cntk

ENV PATH /home/CNTK/build/release/bin:$PATH
