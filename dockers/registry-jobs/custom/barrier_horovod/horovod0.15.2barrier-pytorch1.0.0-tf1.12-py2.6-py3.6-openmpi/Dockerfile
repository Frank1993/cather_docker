FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Labels for the docker
LABEL description="This docker is a special docker that include a special Horovod, this Horovod have been modified that support 2 special function barrier and all_gather_str" \
      repository="philly/jobs/custom/barrier_horovod" \
      tag="horovod0.15.2barrier-pytorch1.0.0-tf1.12-py2.6-py3.6-openmpi" \
      creator="yuanya" \
      tooltype="tensorflow" \
      tooltypeversion="1.12" \
      createtime="14/06/2019"

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu

ENV PATH=$PATH:/usr/local/cuda-9.0/bin:/usr/local/cuda9/bin:/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib/stubs

# install needed software
RUN apt-get -y update && \
    apt-get -y install \
      unzip \
      software-properties-common \
      python-software-properties \
      nano \
      vim \
      joe \
      wget \
      curl \
      jq \
      gawk \
      psmisc \
      python \
      python-dev \
      python-pip \
      python3 \
      python3-dev \
      python3-pip \
      python3-venv  \
      python-six \
      build-essential \
      automake \
      openjdk-8-jdk \
      lsof \
      libcupti-dev \
      # SSH library is necessary for mpi workload.
      openssh-server \
      openssh-client \
      build-essential \
      autotools-dev \
      cmake \
      git \
      bash-completion \
      ca-certificates \
      inotify-tools \
      rsync \
      realpath \
      libjpeg-dev \
      libpng-dev \
      net-tools \
      libsm6 \
      libxext6 \
      rpm \
      #For Azure RDMA and intel MPI installation
      cpio \
      net-tools \
      libdapl2 \
      dapl2-utils \
      libmlx4-1 \
      libmlx5-1 \
      ibutils \
      librdmacm1 \
      libibverbs1 \
      libmthca1 \
      ibverbs-utils \
      rdmacm-utils \
      perftest \
      kmod

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
ENV NCCL_VERSION=2.3.7-1
RUN wget --no-verbose http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl2_${NCCL_VERSION}+cuda9.0_amd64.deb && \
    wget --no-verbose http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl-dev_${NCCL_VERSION}+cuda9.0_amd64.deb && \
    dpkg -i libnccl2_${NCCL_VERSION}+cuda9.0_amd64.deb && \
    dpkg -i libnccl-dev_${NCCL_VERSION}+cuda9.0_amd64.deb && \
    rm libnccl*

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

# install hadoop
ENV HADOOP_VERSION=2.7.6
COPY build_scripts /root/build_scripts
RUN mkdir /tmp/hadoop
RUN cd /tmp/hadoop && \
    wget http://www-us.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz && \
    tar xzf hadoop-${HADOOP_VERSION}.tar.gz -C /usr/local   && \
    ln -sf /usr/local/hadoop-${HADOOP_VERSION} /usr/local/hadoop    &&  \
    rm -rf /usr/local/hadoop/etc/hadoop &&  \
    cp -r /root/build_scripts/hadoop /usr/local/hadoop/etc/hadoop

# update pip
# RUN pip install --upgrade pip && pip install --upgrade setuptools
RUN pip --default-timeout=10000 install --upgrade pip

# Install TensorFlow
ENV TENSORFLOW_VERSION=1.12.0
RUN pip3 --default-timeout=10000 install tensorflow-gpu==${TENSORFLOW_VERSION} h5py && \
    ldconfig

# Install PyTorch
ENV PYTORCH_VERSION=1.0.0
RUN pip3 --default-timeout=10000 install --no-cache-dir torch==${PYTORCH_VERSION} && \
    ldconfig

# Install Dependencies
RUN pip3 --default-timeout=10000 install --no-cache-dir scipy jupyter ipykernel numpy toolz pandas scikit-learn pillow

# install special horovod
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install git-lfs &&  \
    git lfs install &&  \
    mkdir /tmp/HorovodBarrierWheel &&  \
    cd /tmp/HorovodBarrierWheel && \
    git lfs clone https://github.com/amazingyyc/HorovodBarrierWheel.git &&  \
    cd HorovodBarrierWheel && ls -l && \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=0 pip3 --default-timeout=10000 install wheel/horovod-0.15.2-cp35-cp35m-linux_x86_64.whl

# setup appsunner
RUN echo "AppRunner=master@239cc8b0189456e482e9e4854eb7c9045fae0f27"
ENV APPRUNNER_DIR=/tmp/apprunner
ENV APPRUNNER_RUNTIME_BINARY=tensorflow_community_gpu_1.13.1_cp35

RUN mkdir /tmp/apprunner
COPY start_apprunner.py /tmp/apprunner/start_apprunner.py
WORKDIR $APPRUNNER_DIR
RUN python3 start_apprunner.py --host-platform Philly --app-type tensorflow --runtime-binary $APPRUNNER_RUNTIME_BINARY --setup-only && \
    chmod 777 -R .

# Make run_tf global, activate venv
ENV PATH /home/job:$APPRUNNER_DIR/.working/runtime/env/bin:$PATH

RUN mkdir /home/job
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute
WORKDIR /home/job/