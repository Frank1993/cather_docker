FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Labels for the docker
LABEL description="Horovod barrier v3 with openmpi" \
      repository="philly/jobs/custom/barrier_horovod" \
      tag="horovod-v0.15.2-barrier-v3" \
      creator="yuanya" \
      tooltype="tensorflow" \
      tooltypeversion="1.12" \
      createtime="22/06/2019"

ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu

ENV PATH=$PATH:/usr/local/cuda-9.0/bin:/usr/local/cuda9/bin:/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib/stubs

RUN apt-get -y update && \
    apt-get install -y \
        unzip \
        software-properties-common \
        python-software-properties \
        git

# Install Mellanox OFED user-mode drivers and its prereqs
WORKDIR /root
ENV OFED_VERSION=3.3-1.0.4.0

RUN DEBIAN_FRONTEND=noninteractive \
    apt-get -y update && \
    DEBIAN_FRONTEND=noninteractive \
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
        gfortran \
        tk \
        bison \
        flex \
        iproute2 \
        aria2 \
        net-tools \
        openjdk-8-jdk \
        openjdk-8-jre-headless \
        numactl \
        libnuma1 && \
    wget -q -O - http://www.mellanox.com/downloads/ofed/MLNX_OFED-$OFED_VERSION/MLNX_OFED_LINUX-$OFED_VERSION-ubuntu16.04-x86_64.tgz | tar xzf - && \
    cd MLNX_OFED_LINUX-$OFED_VERSION-ubuntu16.04-x86_64/DEBS && \
    for dep in libibverbs1 libibverbs-dev ibverbs-utils libmlx4-1 libmlx5-1 librdmacm1 librdmacm-dev libibumad libibumad-devel libibmad libibmad-devel libopensm infiniband-diags; do \
        dpkg -i $dep\_*_amd64.deb; \
    done && \
    cd ../.. && \
    rm -rf MLNX_OFED_LINUX-*


# Install NCCL
ENV NCCL_VERSION=2.1.15-1
RUN wget --no-verbose http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl2_${NCCL_VERSION}+cuda9.0_amd64.deb && \
    wget --no-verbose http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl-dev_${NCCL_VERSION}+cuda9.0_amd64.deb && \
    dpkg -i libnccl2_${NCCL_VERSION}+cuda9.0_amd64.deb && \
    dpkg -i libnccl-dev_${NCCL_VERSION}+cuda9.0_amd64.deb && \
    rm libnccl*


# Install OpenMPI with CUDA and openib
ENV OPENMPI_VERSIONBASE=1.10
ENV OPENMPI_VERSION=${OPENMPI_VERSIONBASE}.3
ENV OPENMPI_STRING=openmpi-${OPENMPI_VERSION}

WORKDIR $STAGE_DIR

RUN wget -q -O - https://www.open-mpi.org/software/ompi/v${OPENMPI_VERSIONBASE}/downloads/${OPENMPI_STRING}.tar.gz | tar -xzf - && \
    cd ${OPENMPI_STRING} && \
    ./configure --with-cuda --with-openib --prefix=/usr/local/${OPENMPI_STRING} && \
    make -j"$(nproc)" install && \
    rm -rf $STAGE_DIR/${OPENMPI_STRING} && \
    ln -s /usr/local/${OPENMPI_STRING} /usr/local/mpi && \
    # Sanity check:
    test -f /usr/local/mpi/bin/mpic++


# Install Hadoop
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64 \
    HADOOP_PREFIX=/usr/local/hadoop \
    HADOOP_BIN_DIR=/usr/local/hadoop/bin \
    HADOOP_SBIN_DIR=/usr/local/hadoop/sbin \
    HADOOP_COMMON_HOME=/usr/local/hadoop \
    HADOOP_HDFS_HOME=/usr/local/hadoop \
    HADOOP_MAPRED_HOME=/usr/local/hadoop \
    HADOOP_YARN_HOME=/usr/local/hadoop \
    HADOOP_CONF_DIR=/usr/local/hadoop/etc/hadoop \
    HADOOP_ROOT_LOGGER=INFO,console \
    HADOOP_SECURITY_LOGGER=INFO,console \
    YARN_CONF_DIR=$HADOOP_PREFIX/etc/hadoop \
    PATH="/usr/local/hadoop/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/hadoop/lib/native:/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server:${LD_LIBRARY_PATH}"

COPY AppRunner /tmp/AppRunner
COPY build_scripts /root/build_scripts
RUN cd /root/build_scripts && ./install_hadoop.sh && rm -rf /root/build_scripts

# Install Python
ENV PYTHON_WITH_VERSION=python3.5
ENV PIP_WITH_VERSION=pip3.5

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get -y update && \
    apt-get install -y \
        $PYTHON_WITH_VERSION \
        $PYTHON_WITH_VERSION-dev \
        $PYTHON_WITH_VERSION-venv

RUN wget https://bootstrap.pypa.io/get-pip.py && \
    $PYTHON_WITH_VERSION get-pip.py && \
    rm get-pip.py

# Setup AppRunner
RUN echo "AppRunner=master@dd692377347ab0eca0db372df1d0a4cd99b9795b, Runtime=tensorflow_community_gpu_1.12.0_cp35@20181128"
ENV APPRUNNER_DIR=/tmp/apprunner
ENV APPRUNNER_RUNTIME_BINARY=tensorflow_community_gpu_1.12.0_cp35

WORKDIR $APPRUNNER_DIR
RUN cp /tmp/AppRunner/Scripts/start_apprunner.py start_apprunner.py && \
    python3 start_apprunner.py --apprunner-data-root /tmp/AppRunner --host-platform Philly --app-type tensorflow --runtime-binary $APPRUNNER_RUNTIME_BINARY --setup-only && \
    chmod 777 -R .

# Create the home directory for the custom-docker
RUN mkdir /home/job
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Make run_tf global, activate venv
ENV PATH /home/job:$APPRUNNER_DIR/.working/runtime/env/bin:$PATH

# Specify the cudnn version.
ENV CUDNN_VERSION 7.4.1.5
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn7=$CUDNN_VERSION-1+cuda9.0 \
    libcudnn7-dev=$CUDNN_VERSION-1+cuda9.0 && \
    rm -rf /var/lib/apt/lists/*

# set path
RUN ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcublas.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcublas.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so.1 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcufft.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcufft.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcufftw.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcufftw.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcurand.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcurand.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcusolver.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcusolver.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcusparse.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcusparse.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppc.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppc.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppial.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppial.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppicc.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppicc.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppicom.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppicom.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppidei.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppidei.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppif.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppif.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppig.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppig.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppim.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppim.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppist.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppist.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppisu.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppisu.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppitc.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppitc.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnpps.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnpps.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnvgraph.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnvgraph.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnvidia-ml.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnvidia-ml.so.1 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnvrtc.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnvrtc.so.9.0

# update pip
RUN pip3 --default-timeout=10000 install --upgrade pip

# install pytorh
 RUN pip3 install --no-cache-dir torch==1.0.0

# install tensorflow_ranking
RUN pip3 --default-timeout=10000 install tensorflow_ranking==0.1.2

# install horovod
RUN mkdir /tmp/Horovod && \
    cd /tmp/Horovod && \
    git clone --recursive https://github.com/amazingyyc/horovod.git && \
    cd horovod && \
    git checkout -b v0.15.2barrier origin/v0.15.2barrier && \
    python3 setup.py sdist && \ 
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=0 pip3 --default-timeout=10000 install dist/horovod-0.15.2.tar.gz && \
    ldconfig