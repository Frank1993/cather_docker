#cuda:10.0-base-ubuntu16.04
FROM ubuntu:16.04
# Labels for the docker
LABEL description="face_release2" \
      repository="philly/jobs/custom/cntk-msra-face-test" \
      tag="face_release2" \
      creator="v-zhimao" tooltype="cntk" \
      createtime="06/06/2019"
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates apt-transport-https gnupg-curl && \
    rm -rf /var/lib/apt/lists/* && \
    NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
    NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
    apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +5 > cudasign.pub && \
    echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sha256sum -c --strict - && rm cudasign.pub && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
ENV CUDA_VERSION 10.0.130
ENV CUDA_PKG_VERSION 10-0=$CUDA_VERSION-1
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-drivers=410.48-1 && \
    ldconfig -p
# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION \
        cuda-compat-10-0=410.48-1 && \
    ln -s cuda-10.0 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.0 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=410,driver<411"
# cuda:10.0-runtime-ubuntu16.04
ENV NCCL_VERSION 2.4.7
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-libraries-$CUDA_PKG_VERSION \
        cuda-nvtx-$CUDA_PKG_VERSION \
        libnccl2=$NCCL_VERSION-1+cuda10.0 && \
    apt-mark hold libnccl2 && \
    rm -rf /var/lib/apt/lists/*
#cuda:10.0-devel-ubuntu16.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-libraries-dev-$CUDA_PKG_VERSION \
        cuda-nvml-dev-$CUDA_PKG_VERSION \
        cuda-minimal-build-$CUDA_PKG_VERSION \
        cuda-command-line-tools-$CUDA_PKG_VERSION \
        libnccl-dev=$NCCL_VERSION-1+cuda10.0 && \
    rm -rf /var/lib/apt/lists/*
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs
#cuda:10.0-cudnn7-devel-ubuntu16.04
ENV CUDNN_VERSION 7.5.0.56
RUN apt-get update && apt-get install -y --no-install-recommends \
            libcudnn7=$CUDNN_VERSION-1+cuda10.0 \
            libcudnn7-dev=$CUDNN_VERSION-1+cuda10.0 && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*
# Environment variables
ENV STAGE_DIR=/root/gpu/install \
    CUDNN_DIR=/usr/local/cudnn \
    CUDA_DIR=/usr/local/cuda-10.0 \
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
ARG CUB_VERSION=1.8.0
RUN wget https://github.com/NVlabs/cub/archive/${CUB_VERSION}.zip && \
    unzip -d /usr/local ${CUB_VERSION}.zip && \
    rm ${CUB_VERSION}.zip
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
ENV CNTK_ENABLE_ASGD true
RUN chmod -R 777 /home/
RUN cd /home/ && \
    git clone -b release2 https://github.com/Veason-silverbullet/CNTK.git && \
    cd CNTK && \
    git submodule update --init external/gsl && \
    git submodule update --init Source/CNTKv2LibraryDll/proto/onnx/onnx_repo && \
    git submodule update --init Source/Multiverso && \
    mkdir build/release -p && \
    cd build/release && \
    ../../configure --asgd=no --cuda=yes --java=no --python=no --csharp=no && \
    make clean && make -j $(nproc) && \
    cd bin && \
    chmod 777 cntk
ENV PATH /home/CNTK/build/release/bin:$PATH
