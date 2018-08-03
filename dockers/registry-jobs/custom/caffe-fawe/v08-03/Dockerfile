FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.8.0-cudnn.5
# Copy the files to the necessary folder
RUN mkdir /home/job
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod a+rwx /home/job/toolkit-execute

#New-Copy caffe
RUN mkdir /home/caffe
COPY caffe_src.zip /home/caffe/caffe_src.zip
RUN mkdir /home/ncclv1

# Labels for the docker
LABEL description="This docker has BVLC Caffe and python2, add focal loss layer" \
      repository="philly/jobs/custom/caffe-fawe" \
      tag="v08-03" \
      creator="fawe" tooltype="caffe" \
      tooltypeversion="1.0" \
      createtime="08/03/2018"

RUN apt-get -y update
RUN apt-get -y install -y \ 
      python-pip

# upgrade pip
RUN pip install --upgrade pip

## install development requirements
RUN apt-get install -y \
    libprotobuf-dev \
    protobuf-compiler \
    libatlas-base-dev \
    libboost-all-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libhdf5-serial-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libsnappy-dev \
    python-dev \
    vim \
    git \
    tmux \
    cmake \
    libopenblas-dev \
    zip \
    unzip \
    gcc

RUN apt-get autoremove

# Newer NCCL
RUN cd /home/ncclv1 && git clone https://github.com/NVIDIA/nccl.git && cd nccl && \
    git checkout 03d856977ecbaac87e598c0c4bafca96761b9ac7 && \
    make -j$(nproc) && make install

ENV LD_LIBRARY_PATH /home/ncclv1/nccl/build/lib:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/home/ncclv1/nccl/build/lib:${LD_LIBRARY_PATH}

# pip packages
COPY requirements.txt /home/job
RUN pip install -r /home/job/requirements.txt

#New-Make Caffe
RUN chmod -R 777 /home/
RUN unzip -d /home/caffe /home/caffe/caffe_src.zip
RUN cd /home/caffe && make clean && make -j$(nproc) && make -j$(nproc) pycaffe

#Add pycaffe to pythonpath
ENV PYTHONPATH /home/caffe/python:${PYTHONPATH}
ENV PYTHONPATH=/home/caffe/python:${PYTHONPATH}