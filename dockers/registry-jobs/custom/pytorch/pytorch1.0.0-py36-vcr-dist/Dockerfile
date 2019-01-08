FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
# RUN mkdir /home/job
# RUN chmod u+x toolkit-execute
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker pytorch1.0.0 using anaconda python36 with other toolkits" \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch1.0.0-py36-vcr" \
      creator="v-wesu" tooltype="pytorch" \
      tooltypeversion="1.0.0" \
      createtime="1/4/2019"
# Everything above this line is required for the docker.
# Add your personalized features below here.

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

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

# Set CUDA_ROOT
RUN export CUDA_HOME="/usr/local/cuda"

RUN pip install --upgrade pip && \
    pip install EasyDict==1.7 \
                hickle==2.1.0 \
                opencv-python==3.4.4.19 \
                tensorboard==1.0.0a6 \
                shapely==1.6.4 \
                future \
                Cython \ 
                scipy \
                pandas \
                pyyaml \
                json_tricks \
                scikit-image \
                tqdm \
                gputil \
                tensorboardX \
                jsonlines

RUN conda install pytorch==1.0.0 torchvision==0.2.1 -c pytorch
RUN pip install pytorch-pretrained-bert

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
