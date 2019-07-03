FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has custom pytorch in python36 with cuda 9.0, and cudnn 7.0. with mpi nccl" \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch1.1-py36-cuda9-mpi-nccl-hvd" \
      creator="nnn" tooltype="pytorch" \
      tooltypeversion="1.0.0" \
      createtime="07/01/2019"

RUN apt-get update && apt-get install -y --no-install-recommends \
         git \
		 unzip \
         libnccl2=2.1.15-1+cuda9.0 \
         libnccl-dev=2.1.15-1+cuda9.0 \
         ca-certificates \
         libjpeg-dev \
         libpng-dev &&\
         rm -rf /var/lib/apt/lists/*

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    wget \
    libopenblas-dev \
    libopencv-dev && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential

# Set timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

# Get Conda-ified Python.
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh -O ~/anaconda.sh && \
    sh ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

ENV PATH /opt/conda/bin:$PATH

# Install general libraries
RUN conda install -y setuptools Cython

# Install pytorch 
RUN conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

RUN pip install opencv-python
RUN conda update -n base conda
RUN pip install scikit-image
#RUN conda install -y mkl-include cmake cffi typing

# Set CUDA_ROOT
RUN export CUDA_HOME="/usr/local/cuda"

# Install horovod
RUN HOROVOD_GPU_ALLREDUCE=NCCL pip install --no-cache-dir horovod
