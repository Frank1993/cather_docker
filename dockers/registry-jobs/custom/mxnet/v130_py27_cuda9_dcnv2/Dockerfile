FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod a+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker mxnet1.3.1 add FP16 support for some operators." \
      repository="philly/jobs/custom/mxnet" \
      tag="v130_py27_cuda9_dcnv2_rui" \
      creator="v-wesu" tooltype="mxnet" \
      tooltypeversion="1.3.0" \
      createtime="01/02/2019"
# Everything above this line is required for the docker.
# Add your personalized features below here.

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    wget \
    libopenblas-dev \
    libopencv-dev \
    libyaml-dev \
    vim \
    tmux \
    git && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# build from source t
RUN cd ~ && git clone --recursive -b 1.3.0 https://github.com/apache/incubator-mxnet.git
RUN cd ~ && git clone https://github.com/msracver/Deformable-ConvNets.git
RUN cp -r ~/Deformable-ConvNets/DCNv2_op/* ~/incubator-mxnet/src/operator/contrib
RUN cd ~/incubator-mxnet && make -j "$(nproc)" USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1 USE_DIST_KVSTORE=1


RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda2-5.1.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

ENV PATH /opt/conda/bin:$PATH

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
                scikit-image

# Install Python package
RUN cd ~/incubator-mxnet/python && python setup.py install
