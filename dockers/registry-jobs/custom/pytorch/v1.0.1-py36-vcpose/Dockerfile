FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:philly-ubuntu.16.04-cuda.10.0-cudnn.7
# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 1.0.1 in anaconda python36 with cuda 10.0, and cudnn 7.0." \
      repository="philly/jobs/custom/pytorch" \
      tag="v1.0.1-py36-vcpose" \
      creator="bixi" tooltype="pytorch" \
      tooltypeversion="1.0.1" \
      createtime="12/02/2019"
# Everything above this line is required for the docker.
# Add your personalized features below here.

# Setup dependencies, install cudnn, nccl2, python and pip.
ARG NCCL_INSTALL_VERSION=2.3.7-1+cuda10.0
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
        libyaml-dev \
        tmux \
        htop \
        ca-certificates \
        libnccl2=${NCCL_INSTALL_VERSION} \
        libnccl-dev=${NCCL_INSTALL_VERSION} \
        libjpeg-dev \
        libpng-dev

ARG MINICONDA_VERSION=4.5.4
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=$PYTHON_INSTALL_VERSION numpy pyyaml scipy ipython mkl mkl-include cython typing && \
	/opt/conda/bin/conda install -y -c mingfeima mkldnn && \
	/opt/conda/bin/conda install -y -c soumith magma-cuda100 && \
    /opt/conda/bin/conda clean -ya


ENV PATH /opt/conda/bin/:$PATH

# Install general libraries
RUN pip install --upgrade pip && \
    pip install EasyDict==1.7 \
                opencv-python==3.4.1.15 \
                tensorboardX>=1.2 \
                shapely==1.6.4 \
                yacs==0.1.5 \
                Cython \
                scipy \
                pyyaml \
                json_tricks \
                scikit-image \
                mpi4py \
                pandas \
                munkres \
                tqdm

RUN git clone -b v1.0.1 --recursive https://github.com/pytorch/pytorch
RUN cd pytorch; python setup.py install; cd ..

RUN git clone https://github.com/pytorch/vision.git
RUN cd vision; git checkout v0.2.1; python setup.py install; cd ..

# Install horovod
RUN HOROVOD_GPU_ALLREDUCE=NCCL pip install --no-cache-dir horovod

# Install apex
WORKDIR $STAGE_DIR
RUN pip uninstall -y apex || :
# SHA is something the user can touch to force recreation of this Docker layer, 
# and therefore force cloning of the latest version of Apex
RUN SHA=ToUcHMe git clone https://github.com/NVIDIA/apex.git
WORKDIR $STAGE_DIR/apex
RUN python setup.py install --cuda_ext --cpp_ext
WORKDIR $STAGE_DIR
RUN rm -rf apex

WORKDIR /workspace
RUN chmod -R a+w /workspace

