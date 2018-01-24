FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.8.0-cudnn.5
# Copy the files to the necessary folder
RUN mkdir /home/job
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod a+rwx /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has MSR CCS Caffe and python2" \
      repository="philly/jobs/custom/msrccs-caffe" \
      tag="v1.0" \
      creator="ehazar" tooltype="caffe" \
      tooltypeversion="1.0" \
      createtime="1/18/2018"

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
RUN apt-get install -y \
        vim \
        libnccl2=2.1.2-1+cuda8.0 libnccl-dev=2.1.2-1+cuda8.0 \
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
        python-dev

RUN apt-get autoremove

COPY requirements.txt /home/job 
RUN pip install -r /home/job/requirements.txt

# needed scripts
COPY scripts /home/job/

# To build in the docker directly (make sure /src/Caffe has the source):
#RUN mkdir -p /src/Caffe/build
#RUN cd /src/Caffe/build && cmake -DUSE_OPENMP=1 -DUSE_CUDNN=1 -DUSE_NCCL=1 -DCMAKE_INSTALL_PREFIX=$CAFFE_ROOT .. && \
#    make install
# To prebuild .tar.gz file:
# make DESTDIR=/caffe_install install
# tar -zcvf /src/caffe.tar.gz /caffe_install
COPY caffe.tar.gz /home/job
RUN tar --strip-components=1 -C / -xvf /home/job/caffe.tar.gz

# remove intermediates
RUN rm -rf /home/job/*.tar.gz

# To Build:
# docker build -t msrccs-caffe .
# To clean up older images:
#  0. If docker > 1.13: docker system prune
#  1. Stop all exited containers
#   docker rm $(docker ps -a -q -f "status=exited")
#  2. Remove all dangling images (failed builds, ...)
#   docker rmi $(docker images -aq --filter dangling=true)
# To develop on Windows (note the mount: /src):
#  winpty docker run -v d:\\development\\:/src --rm -ti msrccs-caffe
