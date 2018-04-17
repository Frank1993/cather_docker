FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.8.0-cudnn.5
# Copy the files to the necessary folder
RUN mkdir /home/job
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod a+rwx /home/job/toolkit-execute

#New-Copy caffe
RUN mkdir /home/caffe
COPY caffe_src.zip /home/caffe/caffe_src.zip

# Labels for the docker
LABEL description="This docker has BVLC Caffe and python2" \
      repository="philly/jobs/custom/caffe-fawe" \
      tag="rc5-py2" \
      creator="fawe" tooltype="caffe" \
      tooltypeversion="1.0" \
      createtime="4/16/2018"

# sane bash defaults
COPY bashrc.sh /etc/
RUN chmod 655 /etc/bashrc.sh
RUN echo "source /etc/bashrc.sh" >> /etc/bash.bashrc

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

#New-install extra dependence
RUN apt-get install -y libopenblas-dev
RUN apt-get install -y zip
RUN apt-get install -y unzip
RUN apt-get install -y gcc

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

#New-Make Caffe
RUN chmod -R 777 /home/
RUN unzip /home/caffe/caffe_src.zip && cd /home/caffe
RUN make -j$(nproc)