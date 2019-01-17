FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 1.0.0 in anaconda python36 with cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/pytorch" \
      tag="v1.0.0-nccl-py36-vcpose" \
      creator="bixi" tooltype="pytorch" \
      tooltypeversion="1.0.0" \
      createtime="17/01/2019"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get update && apt-get install -y --no-install-recommends \
         git \
		 unzip \
         libnccl2=2.1.15-1+cuda9.0 \
         libnccl-dev=2.1.15-1+cuda9.0 \
         ca-certificates \
         libjpeg-dev \
         libpng-dev &&\
         rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
         git \
         curl \
         vim \
         cmake \
         tmux \
         htop \
         ca-certificates \
         libnccl2=2.1.15-1+cuda9.0 \
         libnccl-dev=2.1.15-1+cuda9.0 \
         libjpeg-dev \
         libyaml-dev \
         libpng-dev &&\
         rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

ENV PYTHON_VERSION=3.6

RUN curl -o ~/anaconda.sh -O  https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh  && \
     chmod +x ~/anaconda.sh && \
     ~/anaconda.sh -b -p /opt/conda && \
     rm ~/anaconda.sh

ENV PATH /opt/conda/bin/:$PATH

# Install general libraries
RUN conda update -n base conda 
RUN conda install -y numpy scipy ipython mkl scikit-learn matplotlib pandas setuptools Cython h5py graphviz
RUN conda clean -ya
RUN conda install mkl-include cmake cffi typing
RUN conda install -c mingfeima mkldnn

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

RUN git clone -b v1.0.0 --recursive https://github.com/pytorch/pytorch
RUN cd pytorch; python setup.py install; cd ..

RUN git clone https://github.com/pytorch/vision.git
RUN cd vision; git checkout v0.2.1; python setup.py install; cd ..

WORKDIR /workspace
RUN chmod -R a+w /workspace

# Set CUDA_ROOT
RUN export CUDA_HOME="/usr/local/cuda"
