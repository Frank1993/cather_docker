FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
# RUN mkdir /home/job
# RUN chmod u+x toolkit-execute
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker pytorch1.0.0 using anaconda python36 allennlp with other toolkits" \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch1.0.0-py36-vcr-allennlp" \
      creator="v-lize" tooltype="pytorch" \
      tooltypeversion="1.0.0" \
      createtime="1/15/2019"
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

# Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        vim \
        tmux \
        unzip \
        htop \
        zsh

# Set timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN conda install pytorch==1.0.0 torchvision==0.2.1 -c pytorch
RUN pip install pytorch-pretrained-bert

RUN pip install numpy pyyaml setuptools cmake cffi tqdm pyyaml scipy ipython mkl mkl-include cython typing h5py pandas nltk spacy numpydoc scikit-learn
RUN conda install jpeg
RUN pip install git+git://github.com/pytorch/vision.git@24577864e92b72f7066e1ed16e978e873e19d13d
RUN cd ~ && git clone https://github.com/Liu-Ze/r2c.git && cd ~/r2c && pip install -r allennlp-requirements.txt
RUN pip install --no-deps allennlp==0.8.0
RUN python -m spacy download en_core_web_sm

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

RUN pip uninstall -y pillow && CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

