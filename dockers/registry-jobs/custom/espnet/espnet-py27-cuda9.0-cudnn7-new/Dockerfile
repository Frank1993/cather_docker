FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has espnet with cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/espnet" \
      tag="espnet-py27-cuda9.0-cudnn7" \
      creator="t-yud" \
      createtime="12/07/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

# Set language
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV CUDA_HOME=/usr/local/cuda

# Set timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

# Install packages
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    git \
    virtualenv \
    python-dev \
    zlib1g-dev \
    sox \
    libtool \
    subversion \
     build-essential \
    apt-utils \
    cmake \
    g++ \
    gcc \
    git \
    curl \
    wget \
    rsync \
    vim \
    less \
    htop \
    jq \
    tmux \
    unzip \
    sox \
    libtool \
    subversion \
    libatlas3-base \
    gawk  \
    virtualenv\
    emacs\
    flac\
    bc
    
WORKDIR /home/
RUN git clone https://github.com/j4ckl1u/espnet.git
RUN cd /home/espnet/tools/ && git clone https://github.com/kaldi-asr/kaldi.git
RUN cd /home/espnet/tools/kaldi/tools/extras && ./install_mkl.sh
RUN cd /home/espnet/tools/ && make all
RUN rm /home/espnet/tools/kaldi-io-for-python/kaldi_io_py.py && ln -s /home/espnet/tools/kaldi-io-for-python/kaldi_io/kaldi_io.py /home/espnet/tools/kaldi-io-for-python/kaldi_io_py.py
RUN cd /home/espnet/tools && . venv/bin/activate && cd kaldi-io-for-python && pip install .
RUN cd /home/espnet/tools && . venv/bin/activate && pip install sentencepiece




