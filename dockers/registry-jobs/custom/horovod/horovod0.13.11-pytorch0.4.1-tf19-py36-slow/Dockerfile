FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Labels for the docker
LABEL description="This docker has horovod-0.13.11 with support for pytorch-0.4.1, tensorflow-1.9.0, python-3.6 and fairseq. It has OpenMPI v1 and needs to be updated to OpenMPI v3" \
      repository="philly/jobs/custom/horovod" \
      tag="horovod0.13.11-pytorch0.4.1-tf19-py36-slow" \
      creator="wigale" \
      tooltype="horovod" \
      tooltypeversion="0.13.11" \
      createtime="8/10/2018"

# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
# Horovod requires both tensorflow and pytorch to be installed
ENV TENSORFLOW_VERSION=1.9.0 \
    PYTORCH_VERSION=0.4.1 \
    CUDNN_VERSION=7.2.1.38-1+cuda9.0 \
    NCCL_VERSION=2.2.13-1+cuda9.0 \
    NCCL_SOCKET_IFNAME=eth0 \
    NCCL_IB_DISABLE=1 \
    NCCL_IB_CUDA_SUPPORT=0 \
    PYTHON_VERSION=3.6 \
    PATH="/usr/local/mpi/bin:${PATH}" \
    LANG=C.UTF-8

# Install cuda 9.0 patch 1
RUN mkdir /tmp/cuda && \
    cd /tmp/cuda && \
    wget -q https://developer.nvidia.com/compute/cuda/9.0/Prod/patches/1/cuda_9.0.176.1_linux-run && \
    cp /usr/local/cuda/version.txt ./ && \
    sh cuda_9.0.176.1_linux-run --silent --accept-eula --installdir=/tmp/cuda && \
    cp -P lib64/* /usr/local/cuda-9.0/lib64/ && \
    rm -rf /usr/local/cuda/lib64/lib*blas.so.9.0.176 && \
    rm -r /tmp/cuda

# Install cuda 9.0 patch 2
RUN mkdir /tmp/cuda && \
    cd /tmp/cuda && \
    wget -q https://developer.nvidia.com/compute/cuda/9.0/Prod/patches/2/cuda_9.0.176.2_linux-run && \
    cp /usr/local/cuda/version.txt ./ && \
    sh cuda_9.0.176.2_linux-run --silent --accept-eula --installdir=/tmp/cuda && \
    cp -P lib64/* /usr/local/cuda-9.0/lib64/ && \
    rm -rf /usr/local/cuda/lib64/lib*blas.so.9.0.282 && \
    rm -r /tmp/cuda

# Install cuda 9.0 patch 3
RUN mkdir /tmp/cuda && \
    cd /tmp/cuda && \
    wget -q https://developer.nvidia.com/compute/cuda/9.0/Prod/patches/3/cuda_9.0.176.3_linux-run && \
    cp /usr/local/cuda/version.txt ./ && \
    sh cuda_9.0.176.3_linux-run --silent --accept-eula --installdir=/tmp/cuda && \
    cp -P lib64/* /usr/local/cuda-9.0/lib64/ && \
    rm -rf /usr/local/cuda/lib64/lib*blas.so.9.0.333 && \
    rm -r /tmp/cuda

# Install cuda 9.0 patch 4
RUN mkdir /tmp/cuda && \
    cd /tmp/cuda && \
    wget -q https://developer.nvidia.com/compute/cuda/9.0/Prod/patches/4/cuda_9.0.176.4_linux-run && \
    cp /usr/local/cuda/version.txt ./ && \
    sh cuda_9.0.176.4_linux-run --silent --accept-eula --installdir=/tmp/cuda && \
    cp -P lib64/* /usr/local/cuda-9.0/lib64/ && \
    rm -rf /usr/local/cuda/lib64/lib*blas.so.9.0.425 && \
    rm -r /tmp/cuda

# Setup dependencies, install cudnn, nccl2, python3.6 and pip. Replace python with python3.6
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        vim \
        wget \
        ca-certificates \
        libcudnn7=${CUDNN_VERSION} \
        libnccl2=${NCCL_VERSION} \
        libnccl-dev=${NCCL_VERSION} \
        libjpeg-dev \
        libpng-dev \
        python3.6 \
        python3.6-dev && \
    rm /usr/bin/python && \
    ln -s /usr/bin/python3.6 /usr/bin/python && \
    curl -O https://bootstrap.pypa.io/get-pip.py && \
        python get-pip.py && \
        rm get-pip.py

# Install TensorFlow and Keras
RUN pip install tensorflow-gpu==${TENSORFLOW_VERSION} keras h5py

# Install PyTorch
RUN pip install http://download.pytorch.org/whl/cu90/torch-${PYTORCH_VERSION}-cp36-cp36m-linux_x86_64.whl && \
    pip install torchvision

# Install Horovod, temporarily using CUDA stubs
RUN apt-get install libxml2-dev libxslt-dev python3.6-dev -y && \
    ldconfig /usr/local/cuda-9.0/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod && \
    ldconfig

# Install LM dependencies
RUN pip install cffi tqdm spacy mpi4py ipdb pandas matplotlib py3nvml && \
    python -m spacy download en && \
    git clone https://github.com/pytorch/fairseq.git && \
    cd fairseq && \
    python setup.py build && \
    python setup.py develop

# Install AM Team dependencies (separate docker until requirements.txt is supported)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libsndfile-dev && \
    pip install editdistance tensorboard_logger librosa SoundFile && \
    pip install jupyter && \
    pip install https://github.com/kpu/kenlm/archive/master.zip

# Configure NCCL and OpenMPI to run good defaults:
RUN echo "hwloc_base_binding_policy = none" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "rmaps_base_mapping_policy = slot" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "btl = self,sm,tcp" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "btl_tcp_if_exclude = docker0,lo,ib0,ib1" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo NCCL_DEBUG=INFO >> /etc/nccl.conf && \
    echo NCCL_SOCKET_IFNAME=^docker0 >> /etc/nccl.conf

COPY toolkit-execute /home/job/toolkit-execute
RUN chmod a+x /home/job/toolkit-execute
WORKDIR /home/job/
