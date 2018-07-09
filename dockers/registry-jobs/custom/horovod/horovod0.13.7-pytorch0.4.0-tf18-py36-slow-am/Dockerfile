FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Labels for the docker
LABEL description="This docker has horovod-0.13.2 with support for pytorch-0.4, tensorflow-18, python-3.6. It does not include the correct version of OpenMPI, making it inefficient. It contains the required AM team dependencies (editdistance, tensorboard_logger, librosa and SoundFile)" \
      repository="philly/jobs/custom/horovod" \
      tag="horovod0.13.7-pytorch0.4.0-tf18-py36-slow-am" \
      creator="wigale" \
      tooltype="horovod" \
      tooltypeversion="0.13.7" \
      createtime="7/9/2018"

# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
# Horovod requires both tensorflow and pytorch to be installed
ENV TENSORFLOW_VERSION=1.8.0 \
    PYTORCH_VERSION=0.4.0 \
    CUDNN_VERSION=7.0.5.15-1+cuda9.0 \
    NCCL_VERSION=2.2.12-1+cuda9.0 \
    NCCL_SOCKET_IFNAME=eth0 \
    NCCL_IB_DISABLE=1 \
    PYTHON_VERSION=3.6 \
    PATH="/usr/local/mpi/bin:${PATH}"

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
    pip install torchvision && \
    pip install py3nvml

# Install Horovod, temporarily using CUDA stubs
RUN apt-get install libxml2-dev libxslt-dev python3.6-dev -y && \
    ldconfig /usr/local/cuda-9.0/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod && \
    ldconfig

# Install AM Team dependencies (separate docker until requirements.txt is supported)
RUN apt-get install -y --no-install-recommends libsndfile-dev && \
    pip install editdistance tensorboard_logger librosa SoundFile

# Configure NCCL and OpenMPI to run good defaults:
#   --bind-to none --map-by slot --mca btl_tcp_if_exclude lo,docker0
RUN echo "hwloc_base_binding_policy = none" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "rmaps_base_mapping_policy = slot" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "btl = self,sm,tcp" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "btl_tcp_if_exclude = docker0,lo,ib0,ib1" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo NCCL_DEBUG=INFO >> /etc/nccl.conf && \
    echo NCCL_SOCKET_IFNAME=^docker0 >> /etc/nccl.conf

COPY toolkit-execute /home/job/toolkit-execute
COPY barrier.py /home/job/barrier.py
RUN chmod a+x /home/job/toolkit-execute
WORKDIR /home/job/