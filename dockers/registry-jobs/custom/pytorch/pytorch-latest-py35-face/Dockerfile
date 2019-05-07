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

RUN /opt/conda/bin/conda install -y -n $PYTHON_ENV mock opencv six future protobuf && \ 
    /opt/conda/bin/conda clean -ya
RUN git clone https://github.com/facebookresearch/detectron ~/detectron
RUN cd ~/detectron && make
RUN rm -r ~/detectron

WORKDIR /workspace
RUN chmod -R a+w /workspace
