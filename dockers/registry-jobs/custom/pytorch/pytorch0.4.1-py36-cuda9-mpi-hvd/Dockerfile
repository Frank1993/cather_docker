FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:philly-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

RUN rm -rf /usr/local/openmpi-1.10.3
# Install OPENMPI4.0.0
COPY openmpi-4.0.0.tar.gz /
RUN tar -xvf /openmpi-4.0.0.tar.gz -C / && \
    cd /openmpi-4.0.0 && \
    ./configure --prefix=/usr/local/openmpi/ && \
    make -j8 && \
    make install -j8 && \
    # Sanity check:
    test -f /usr/local/openmpi/bin/mpic++

# Update environment variables
ENV PATH=/usr/local/openmpi/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/openmpi/lib:/usr/local/openmpi/lib64:$LD_LIBRARY_PATH

# Labels for the docker
LABEL description="This docker has custom pytorch in python36 with cuda 9.0, and cudnn 7.0. with mpi nccl" \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch1.0.0-py36-mpi-hvd" \
      creator="v-kairli" tooltype="pytorch" \
      tooltypeversion="0.4.1" \
      createtime="18/12/2018"

#RUN wget --no-verbose http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl2_2.1.15-1+cuda9.0_amd64.deb && \
#    dpkg -i libnccl2_2.1.15-1+cuda9.0_amd64.deb

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
    libopencv-dev \
    libyaml-dev \
    git && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        vim \
        tmux \
        unzip \
        htop

# Set timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

# Get Conda-ified Python.
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    sh ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

ENV PATH /opt/conda/bin:$PATH

# Install general libraries
RUN conda install -y python=3.6 numpy pyyaml scipy ipython mkl scikit-learn matplotlib pandas setuptools Cython h5py graphviz
RUN conda clean -ya
RUN conda install -y mkl-include cmake cffi typing
RUN conda install -y -c mingfeima mkldnn


# Install pytorch
RUN conda install -y pytorch==0.4.1 torchvision -c pytorch

# Set CUDA_ROOT
RUN export CUDA_HOME="/usr/local/cuda"

# Install horovod
RUN HOROVOD_GPU_ALLREDUCE=NCCL pip install --no-cache-dir horovod==0.16.4
