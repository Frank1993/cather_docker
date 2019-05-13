FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch latest commit in Anaconda python3.5 with cuda 9.2" \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch-latest-py35-face" \
      creator="v-ximing" \
      tooltype="pytorch" \
      tooltypeversion="latest" \
      createtime="05/06/2019"
# Everything above this line is required for the docker.

# Add your personalized features below here.

# Environment variables
ENV STAGE_DIR=/root/gpu/install \
    PYTHON_VERSION=3.5 \
    PYTHON_ENV=py35 \
    OPENMPI_VERSION=4.0.1 \
    LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH} \
    LANG=C.UTF-8

RUN mkdir -p $STAGE_DIR

RUN apt-get update && \
    apt-get install -y \
      apt-utils \
      software-properties-common \
      autotools-dev \
      htop \
      tree \
      vim \
      build-essential \
      cmake \
      g++-5 \
      gcc-5 \
      git \
      ca-certificates \
      rsync \
      curl \
      wget \
      jq \
      openssh-server \
      openssh-client \
    # No longer in 'minimal set of packages'      
      sudo \
    # ifconfig
      net-tools \
      ca-certificates \
      libjpeg-dev \
      libpng-dev \
      ninja-build && \
    apt-get autoremove && \
    apt-get autoclean

WORKDIR $STAGE_DIR

RUN curl -o ~/miniconda.sh -O  \
        https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda create -y -n $PYTHON_ENV python=${PYTHON_VERSION} \
        numpy pyyaml scipy ipython mkl mkl-include matplotlib setuptools \
        cmake cffi typing cython pillow && \
    /opt/conda/bin/conda clean -ya

ENV PATH=/opt/conda/envs/$PYTHON_ENV/bin:$PATH
ENV CMAKE_PREFIX_PATH=/opt/conda/envs/$PYTHON_ENV

RUN git clone --recursive -b v1.1.0 --single-branch https://github.com/pytorch/pytorch ~/pytorch
RUN cd ~/pytorch && python setup.py install
RUN rm -r ~/pytorch

RUN git clone https://github.com/pytorch/vision.git ~/torchvision
RUN cd ~/torchvision && python setup.py install
RUN rm -r ~/torchvision

RUN git clone https://github.com/cocodataset/cocoapi.git ~/cocoapi
RUN cd ~/cocoapi/PythonAPI && make install
RUN rm -r ~/cocoapi

# OpenMPI
ENV PATH=/usr/local/mpi/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/mpi/lib:/usr/local/mpi/lib64:$LD_LIBRARY_PATH
RUN wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-${OPENMPI_VERSION}.tar.gz -P ~/openmpi
RUN cd ~/openmpi && gunzip -c openmpi-${OPENMPI_VERSION}.tar.gz | tar xf -
RUN cd ~/openmpi/openmpi-${OPENMPI_VERSION} && \
    ./configure --enable-orterun-prefix-by-default --prefix=/usr/local/mpi && \
    make -j$(nproc) all && \
    make install && \
    ldconfig
RUN rm -r ~/openmpi

# Install Horovod, temporarily using CUDA stubs
RUN ldconfig /usr/local/cuda-9.2/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_PYTORCH=1 \
        /opt/conda/envs/$PYTHON_ENV/bin/pip install --no-cache-dir horovod && \
    ldconfig

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

WORKDIR /workspace
RUN chmod -R a+w /workspace
