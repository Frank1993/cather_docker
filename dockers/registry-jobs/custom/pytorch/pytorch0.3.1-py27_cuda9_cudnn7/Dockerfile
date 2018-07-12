FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker contains PyTorch 0.3 using python 2.7 from anaconda with cuda 9.0 and cudnn 7.0 " \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch0.3.1-py27_cuda9_cudnn7" \
      creator="wuzhiron" tooltype="pytorch" \
      tooltypeversion="0.4.0" \
      createtime="7/12/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get update && apt-get install -y --no-install-recommends \
         git \
         curl \
         vim \
         cmake \
         tmux \
         htop \
         ca-certificates \
         libnccl2 \
         libnccl-dev \
         libjpeg-dev \
         libyaml-dev \
         libpng-dev &&\
         rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Install Anaconda for python 2.7
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

# Install general libraries
RUN conda install -y python=2.7 numpy pyyaml scipy ipython mkl scikit-learn matplotlib pandas setuptools Cython h5py cmake lmdb && conda clean -ya

# Install pytorch GPU version.
RUN conda install pytorch=0.3.1 torchvision cuda90 -c pytorch && conda clean -ya

RUN pip install --upgrade pip && \
    pip install EasyDict==1.7 \
                opencv-python==3.4.1.15 \
                tensorboardX \
                shapely==1.6.4 \
                Cython \
                scipy \
                pyyaml \
                json_tricks \
                scikit-image


# Set CUDA_ROOT
ENV CUDA_ROOT /usr/local/cuda/bin
RUN export CUDA_HOME="/usr/local/cuda"

WORKDIR /workspace
RUN chmod -R a+w /workspace
