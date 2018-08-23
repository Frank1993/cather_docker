FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Labels for the docker
ARG PYTHON_VERSION=36
ARG PYTORCH_VERSION=0.4.1
LABEL description="This docker builds the v$PYTORCH_VERSION branch from internal vsts pytorch repo, python." \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch$PYTORCH_VERSION-vsts-py$PYTHON_VERSION" \
      creator="tix" \
      tooltype="pytorch" \
      tooltypeversion="$PYTORCH_VERSION" \
      createtime="8/7/2018"

ENV PATH="/usr/local/mpi/bin:${PATH}"

# Setup dependencies, install cudnn, nccl2, python and pip.
ARG CUDNN_INTALL_VERSION=7.0.5.15-1+cuda9.0
ARG NCCL_INSTALL_VERSION=2.2.12-1+cuda9.0
ARG PYTHON_INSTALL_VERSION=3.6
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
        libcudnn7=${CUDNN_INTALL_VERSION} \
        libcudnn7-dev=${CUDNN_INTALL_VERSION} \
        libnccl2=${NCCL_INSTALL_VERSION} \
        libnccl-dev=${NCCL_INSTALL_VERSION} \
        libjpeg-dev \
        libpng-dev \
        python$PYTHON_INSTALL_VERSION \
        python$PYTHON_INSTALL_VERSION-dev && \
	    rm /usr/bin/python && \
	    ln -s /usr/bin/python$PYTHON_INSTALL_VERSION /usr/bin/python && \
	    curl -O https://bootstrap.pypa.io/get-pip.py && \
        python get-pip.py && \
        rm get-pip.py

ARG MINICONDA_VERSION=4.5.4
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=$PYTHON_INSTALL_VERSION numpy pyyaml scipy ipython mkl mkl-include cython typing && \
    /opt/conda/bin/conda install -y -c pytorch magma-cuda90 && \
	/opt/conda/bin/conda install -y -c mingfeima mkldnn && \
    /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH

# Install pytorch from wheel. only supporting 0.4.0 and 0.4.1 now
RUN pip install https://pytorch.blob.core.windows.net/pytorch-wheel/torch-$PYTORCH_VERSION-cp$PYTHON_VERSION-cp${PYTHON_VERSION}m-linux_x86_64.whl

RUN git clone https://github.com/pytorch/vision.git && cd vision && pip install -v .

COPY toolkit-execute /home/job/toolkit-execute
RUN chmod a+x /home/job/toolkit-execute
WORKDIR /home/job/