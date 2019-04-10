FROM phillyregistry.azurecr.io/microsoft_pytorch:v1.0.1_gpu_cuda9.0_py36_release_gpuenv_hvd0.15.2 

# Labels for the docker
LABEL description="Contains tools for ASR training w/ Pytorch+Kaldi. \
		CUDA 9.0, python 3.6, pytorch 1.0.1, horovod 0.15.2 (from philly image), Kaldi 5.5, kaldi_io (latest from github master branch)" \
      repository="philly/jobs/custom/asr-pytorch-kaldi" \
      tag="philly_1.0.1-kaldi_master" \
      creator="akmahaja" \
      tooltype="asr" \
      tooltypeversion="5.5" \
      createtime="4/8/2019"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common && \
    apt-get update && apt-get install -y --no-install-recommends \
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
    gawk  

RUN pip install --upgrade pip && \
    pip install matplotlib>=2.1.0 && \
    pip install scipy>=1.0.0 && \
    pip install numpy>=1.14.2 && \
    pip install blockdiag>=1.0 && \
    pip install pandas>=0.23.0 

WORKDIR /home

RUN git clone https://github.com/kaldi-asr/kaldi.git && \
    cd kaldi/tools && mkdir -p python && touch python/.use_default_python && \
    ./extras/check_dependencies.sh && \
    ./extras/install_mkl.sh && \
    make -j `nproc` && \
    wget https://sramdevstorage.blob.core.windows.net/public-shared/srilm.tgz && \
    ./extras/install_irstlm.sh && \
    ./extras/install_srilm.sh && \
    cd ../src && ./configure --shared && \
    make depend -j `nproc` && make -j `nproc` && cd ../.. && \
    git clone https://github.com/vesis84/kaldi-io-for-python.git kaldi_io

# Environment variables for kaldi and kaldi_io setup 
ENV KALDI_ROOT=/home/kaldi \
    LC_ALL=C \
    PYTHONPATH=/home/kaldi_io:$PYTHONPATH 

COPY toolkit-execute /home/job/toolkit-execute
RUN chmod a+x /home/job/toolkit-execute
WORKDIR /home/job
