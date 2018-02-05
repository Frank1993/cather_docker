FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.8.0-cudnn.5
# Copy the files to the necessary folder
RUN mkdir /home/job
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod a+rwx /home/job/toolkit-execute
COPY envs.sh /home/job/envs.sh

# Labels for the docker
LABEL description="This docker has BVLC Caffe and python2" \
      repository="philly/jobs/custom/caffe" \
      tag="rc5-py2" \
      creator="ehazar" tooltype="caffe" \
      tooltypeversion="1.0" \
      createtime="2/1/2018"

# sane bash defaults
COPY bashrc.sh /etc/
RUN chmod 655 /etc/bashrc.sh
RUN echo "source /etc/bashrc.sh" >> /etc/bash.bashrc

# Prepare Caffe environment
ENV CAFFE_ROOT=/opt/caffe
RUN mkdir -p $CAFFE_ROOT/lib

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
RUN mkdir -p $PYCAFFE_ROOT

ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/bin:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/lib" >> /etc/ld.so.conf.d/caffe.conf

RUN apt-get -y update
RUN apt-get -y install -y \ 
      python-pip

# upgrade pip
RUN pip install --upgrade pip

## install development requirements
RUN apt-get install -y libprotobuf-dev protobuf-compiler
RUN apt-get install -y libatlas-base-dev libboost-all-dev
RUN apt-get install -y libgflags-dev libgoogle-glog-dev
RUN apt-get install -y libhdf5-serial-dev
RUN apt-get install -y libleveldb-dev liblmdb-dev
RUN apt-get install -y libopencv-dev
RUN apt-get install -y libsnappy-dev
RUN apt-get install -y python-dev
RUN apt-get install -y vim
RUN apt-get install -y git
RUN apt-get install -y tmux
RUN apt-get install -y cmake

RUN apt-get autoremove

# Newer NCCL
RUN git clone https://github.com/NVIDIA/nccl.git && cd nccl && \
    git checkout 03d856977ecbaac87e598c0c4bafca96761b9ac7 && \
    make -j$(nproc) && make install && cd .. && rm -rf nccl

#tmux conf
COPY tmux.conf /etc/

# pip packages
COPY requirements.txt /home/job
RUN pip install -r /home/job/requirements.txt

# example scripts
COPY scripts /home/job/

# To build in the docker directly (make sure /src/caffe has the source):
RUN mkdir -p /src/
RUN cd /src/ && \
    git clone https://github.com/BVLC/caffe && cd caffe && \
    git checkout rc5 && \
    mkdir build && cd build && \
    cmake -DUSE_OPENMP=1 -DUSE_CUDNN=1 -DUSE_NCCL=1 -DCUDA_ARCH_NAME=All -DCMAKE_INSTALL_PREFIX=$CAFFE_ROOT .. && \
    make -j$(nproc) && make install && cd /src && rm -rf caffe
# To prebuild .tar.gz file (of any other caffe):
# make DESTDIR=/caffe_install install
# tar -zcvf /src/caffe.tar.gz /caffe_install
#COPY caffe.tar.gz /home/job
#RUN tar --strip-components=1 -C / -xvf /home/job/caffe.tar.gz

# Make it possible to modify this if needed
RUN chmod -R 777 /opt/

# remove intermediates
RUN rm -rf /home/job/*.tar.gz

# To Build:
# docker build -t caffe .
# To clean up older images:
#  0. If docker > 1.13: docker system prune
#  1. Stop all exited containers
#   docker rm $(docker ps -a -q -f "status=exited")
#  2. Remove all dangling images (failed builds, ...)
#   docker rmi $(docker images -aq --filter dangling=true)
# To develop on Windows (note the mount: /src):
#  winpty docker run -v d:\\development\\:/src --rm -ti caffe
