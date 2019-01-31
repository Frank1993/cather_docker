FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Labels for the docker
LABEL description="This docker has tensorflow-1.10 and python-3.6 and bilm-tf, allennlp" \
      repository="philly/jobs/custom/tf10_py36_allennlp_bilm" \
      tag="tf10_py36_allennlp_bilm" \
      creator="chuah" tooltype="tf10_py36_allennlp_bilm" \
      tooltypeversion="1.10" \
      createtime="01/07/2019"


ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

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

COPY build_scripts /root/build_scripts
RUN cd /root/build_scripts && ./install_hadoop.sh && rm -rf /root/build_scripts


# Install Python
ENV PYTHON_WITH_VERSION=python3.6
ENV PIP_WITH_VERSION=pip3.6

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
RUN echo "AppRunner=master@409e7f2f, Runtime=tensorflow_community_gpu_1.10.0_cp36"
ENV APPRUNNER_DIR=/tmp/apprunner
ENV APPRUNNER_RUNTIME_BINARY=tensorflow_community_gpu_1.10.0_cp36

WORKDIR $APPRUNNER_DIR
RUN curl https://cosmos08.osdinfra.net/cosmos/cosmos.upload/shares/bingads.data.clickpredict/data/ClickPrediction/TrainStation/AppRunner/Scripts/start_apprunner.py -o start_apprunner.py && \
    python3 start_apprunner.py --host-platform Philly --app-type tensorflow --runtime-binary $APPRUNNER_RUNTIME_BINARY --setup-only && \
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

# Install bilm
RUN cd /home/job/; git clone https://github.com/allenai/bilm-tf; cd bilm-tf; python3.6 setup.py install

# Install allen NLP
RUN python3.6 -m pip install allennlp
