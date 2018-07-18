FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
# RUN mkdir /home/job
# RUN chmod u+x toolkit-execute
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker mxnet1.1.0 using anaconda python27 with other toolkits" \
      repository="philly/jobs/custom/mxnet" \
      tag="v110_py27_cuda9" \
      creator="v-jiarxu" tooltype="mxnet" \
      tooltypeversion="1.1.0" \
      createtime="7/18/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    wget \
    libopenblas-dev \
    libopencv-dev \
    libyaml-dev \
    git && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# v-jiarxu: Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        vim \
        tmux \
        htop 

# v-jiarxu: Set timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda2-5.2.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    sh ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN pip install --upgrade pip && \
    pip install mxnet-cu90==1.1.0 \
                EasyDict==1.7 \
                hickle==2.1.0 \
                opencv-python==3.2.0.6 \
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
                gputil 
