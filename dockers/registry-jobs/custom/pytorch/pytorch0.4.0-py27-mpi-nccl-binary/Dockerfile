FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has custom pytorch in python27 with cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch0.4.0-py27-mpi-nccl-binary" \
      creator="vison" tooltype="pytorch" \
      tooltypeversion="0.4.0" \
      createtime="9/26/2018"

COPY nccl-repo-ubuntu1604-2.1.15-ga-cuda9.0_1-1_amd64.deb /home/job/
RUN dpkg -i /home/job/nccl-repo-ubuntu1604-2.1.15-ga-cuda9.0_1-1_amd64.deb

RUN apt-get update && apt-get install -y --no-install-recommends \
         git \
		 unzip \
         libnccl2=2.1.15-1+cuda9.0 \
         libnccl-dev=2.1.15-1+cuda9.0 \
         ca-certificates \
         libjpeg-dev \
         libpng-dev &&\
         rm -rf /var/lib/apt/lists/*

# Get Conda-ified Python.
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

# Install general libraries
RUN conda install -y python=2.7 numpy pyyaml scipy ipython mkl scikit-learn matplotlib pandas setuptools Cython h5py graphviz
RUN conda clean -ya
RUN conda install mkl-include cmake cffi typing
RUN conda install -c mingfeima mkldnn

# pip packages
COPY requirements.txt /home/job
RUN /opt/conda/bin/pip install -r /home/job/requirements.txt

# ...get PyTorch source and build.
RUN git clone -b v0.4.0 --recursive https://github.com/pytorch/pytorch
RUN cd pytorch; python setup.py install; cd ..

# Set CUDA_ROOT
RUN export CUDA_HOME="/usr/local/cuda"
