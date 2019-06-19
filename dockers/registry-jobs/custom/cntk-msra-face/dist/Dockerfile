FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Labels for the docker
LABEL description="dist" \
      repository="philly/jobs/custom/cntk-msra-face" \
      tag="dist" \
      creator="v-zhimao" tooltype="cntk" \
      createtime="06/19/2019"

RUN mkdir /home/job
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

RUN DEBIAN_FRONTEND=noninteractive && \
        apt-get update && apt-get install -y --no-install-recommends \
        autotools-dev \
        build-essential \
        cmake \
        git ca-certificates \
        libavcodec-dev \
        libavformat-dev \
        libjasper-dev \
        libjpeg-dev \
        libpng-dev \
        # TODO Note, selecting 'libpng12-dev' instead of 'libpng-dev'
        liblapacke-dev \
        libswscale-dev \
        libtiff-dev \
        # TODO Note, selecting 'libtiff5-dev' instead of 'libtiff-dev'
        pkg-config \
        wget \
        zlib1g-dev \
        ca-certificates \
        curl \
        unzip \
        automake \
        libtool-bin \
        autoconf \
        libbz2-dev \
        openjdk-8-jdk \
        gawk \
        python \
        python-yaml \
        python-six \
        rsync \
        libpcre++-dev \
        gfortran \
        python-dev \
        automake \
        libtool-bin \
        autoconf \
        subversion \
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


ARG CUB_VERSION=1.7.4
RUN wget https://github.com/NVlabs/cub/archive/${CUB_VERSION}.zip && \
    unzip -d /usr/local ${CUB_VERSION}.zip && \
    rm ${CUB_VERSION}.zip


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
        gfortran \
        tk \
        bison \
        flex \
        libnuma1 \
        checkinstall \
        && \
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


ARG OPENMPI_VERSION=1.10.7
RUN wget -q -O - https://www.open-mpi.org/software/ompi/v1.10/downloads/openmpi-${OPENMPI_VERSION}.tar.gz | tar -xzf - && \
    cd openmpi-${OPENMPI_VERSION} && \
    apt-get -y update && \
    apt-get -y -f install && \
    apt-get -y install libsysfs2 libsysfs-dev && \
    ./configure --with-verbs --prefix=/usr/local/mpi && \
    make -j $(nproc) install && \
    cd .. && \
    rm -rf openmpi-${OPENMPI_VERSION}
ENV PATH /usr/local/mpi/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/mpi/lib:$LD_LIBRARY_PATH


ARG MKLML_VERSION=v0.14/mklml_lnx_2018.0.3.20180406
RUN mkdir /usr/local/mklml && \
    wget --no-verbose -O - https://github.com/01org/mkl-dnn/releases/download/$MKLML_VERSION.tgz | \
    tar -xzf - -C /usr/local/mklml


ARG HALIDE_RELEASE=2018_02_15
ARG HALIDE_RELEASE_GUID=46d8e9e0cdae456489f1eddfd6d829956fc3c843
RUN wget --no-verbose -O - https://github.com/halide/Halide/releases/download/release_$HALIDE_RELEASE/halide-linux-64-gcc53-trunk-$HALIDE_RELEASE_GUID.tgz | tar -xzf - -C /usr/local
ENV PATH /usr/local/halide/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/halide/lib:$LD_LIBRARY_PATH
ENV HALIDE_PATH /usr/local/halide


ARG NCCL_VERSION=2.4.7
RUN wget --no-verbose http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64/libnccl2_2.4.7-1+cuda9.0_amd64.deb && \
    dpkg -i libnccl2_2.4.7-1+cuda9.0_amd64.deb && \
    rm libnccl*


ENV CNTK_ENABLE_ASGD true
RUN chmod -R 777 /home/
RUN cd /home/ && \
    git clone -b lewlu/msra-face-dist https://github.com/Microsoft/CNTK.git && \
    cd CNTK && \
    git submodule update --init external/gsl && \
    git submodule update --init Source/CNTKv2LibraryDll/proto/onnx/onnx_repo && \
    git submodule update --init Source/Multiverso && \
    mkdir build/release -p && \
    cd build/release && \
    ../../configure --asgd=no && \
    make clean && make -j $(nproc) && \
    cd bin && \
    chmod 777 cntk
ENV PATH /home/CNTK/build/release/bin:$PATH

