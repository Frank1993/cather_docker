FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.8.0-cudnn.5

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="caffe-lstm-ctc" \
      repository="philly/jobs/custom/caffe-speech" \
      tag="v07-25" \
      creator="v-haidin" tooltype="caffe-speech" \
      tooltypeversion="1.0" \
      createtime="2/28/2018"

# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get -y update

## general dependenvies
RUN apt-get -y install \
      libprotobuf-dev \ 
      libleveldb-dev \ 
      libsnappy-dev \ 
      libopencv-dev \ 
      libhdf5-serial-dev \ 
      protobuf-compiler \ 
      libgflags-dev \ 
      libgoogle-glog-dev \ 
      liblmdb-dev \ 
      git \ 
      zip \ 
      python-pip \ 
      gfortran && \ 
    apt-get -y --no-install-recommends install libboost-all-dev
RUN apt-get -y install git
RUN pip install numpy

## Note: there exists an older version of opemmpi in /usr/lib, re-link it to the version installed
RUN ln -sfn /usr/local/openmpi-1.10.3/lib/libmpi.so.12 /usr/lib/libmpi.so.1
RUN ln -sfn /usr/local/openmpi-1.10.3/lib/libmpi.so.12 /usr/lib/libmpi.so
RUN ln -sfn /usr/local/openmpi-1.10.3/lib/libmpi_cxx.so.1 /usr/lib/libmpi_cxx.so.1
RUN ln -sfn /usr/local/openmpi-1.10.3/lib/libmpi_cxx.so.1 /usr/lib/libmpi_cxx.so

## NCCL 
RUN wget --no-verbose http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl2_2.2.13-1+cuda8.0_amd64.deb && \
	apt-get -y update && \
	apt-get -y install libnccl2=2.2.13-1+cuda8.0 libnccl-dev=2.2.13-1+cuda8.0 && \
    rm libnccl*
	
# BUILDING CAFFE
ENV CAFFE_DIR=/root/caffe
RUN mkdir -p $CAFFE_DIR

RUN mkdir -p $CAFFE_DIR/openblas
WORKDIR $CAFFE_DIR/openblas
## openblas
RUN git clone git://github.com/xianyi/OpenBLAS
WORKDIR $CAFFE_DIR/openblas/OpenBLAS
RUN make FC=gfortran
RUN make install

RUN mkdir -p /usr/local/OpenBLAS
RUN cp -r /opt/OpenBLAS/lib /usr/local/OpenBLAS
ENV LD_LIBRARY_PATH=/usr/local/OpenBLAS/lib:$LD_LIBRARY_PATH

# CAFFE
WORKDIR $CAFFE_DIR
COPY /Caffe-TrainingTool.zip $CAFFE_DIR

RUN unzip $CAFFE_DIR/Caffe-TrainingTool.zip
WORKDIR $CAFFE_DIR/Caffe-TrainingTool/caffe
RUN cp Makefile.config.16.04.example Makefile.config
RUN make all -j8

RUN make distribute

RUN mkdir -p /usr/local/caffe/bin
RUN mkdir -p /usr/local/caffe/lib
RUN cp -r $CAFFE_DIR/Caffe-TrainingTool/caffe/distribute_linux/bin /usr/local/caffe
RUN cp -r $CAFFE_DIR/Caffe-TrainingTool/caffe/distribute_linux/lib /usr/local/caffe

# RUN ls /opt/caffe/bin
# RUN ls /opt/caffe/lib

# Update environment variables
ENV PATH=/usr/local/caffe/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/caffe/lib:$LD_LIBRARY_PATH

# RUN echo $PATH
# RUN echo "PATH=$PATH" >> ~/.ssh/environment
# RUN echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> ~/.ssh/environment

# Suppress libdc1394 error
RUN ln -s /dev/null /dev/raw1394
RUN echo 'ln -s /dev/null /dev/raw1394' >> ~/.bashrc

# Copy toolkit-execute to /home/custom/runCust in order to run at Philly@AP
RUN mkdir /home/custom
COPY /toolkit-execute /home/custom/runCust
RUN chmod u+x /home/custom/runCust
