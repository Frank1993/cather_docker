FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.14.04-cuda.8.0-cudnn.5

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="caffe-iu" \
      repository="philly/jobs/custom/caffe-iu" \
      tag="v1.0" \
      creator="jiaphuan" tooltype="caffe-iu" \
      tooltypeversion="1.0" \
      createtime="4/24/2018"

# Everything above this line is required for the docker.
# Add your personalized features below here.

# Get base dependecies if not there
RUN apt-get -y update && \
    apt-get -y install software-properties-common && \
    apt-get -y install python3-software-properties && \
    apt-get -y install python-software-properties && \
    add-apt-repository ppa:openjdk-r/ppa && \
    apt-get -y update && \
    apt-get -y install openjdk-7-jdk  

RUN apt-get -y update && \
    apt-get -y install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler && \
    apt-get -y install --no-install-recommends libboost-all-dev && \
    apt-get -y install libgflags-dev libgoogle-glog-dev liblmdb-dev && \
    apt-get -y install cmake git python-dev python-numpy python-pip python-scipy && \
    apt-get -y install libatlas-base-dev 

RUN apt-get -y install bc zlib1g-dev
RUN pip install packaging appdirs

RUN apt-get -y install unzip

RUN apt-get update
RUN apt-get -y install cuda-toolkit-8.0 --fix-missing
RUN apt-get -y install libcudnn5-dev

WORKDIR /home/custom
RUN wget http://nih.at/libzip/libzip-1.1.2.tar.gz
RUN tar -xzvf ./libzip-1.1.2.tar.gz
WORKDIR /home/custom/libzip-1.1.2
RUN ./configure
RUN make -j all
RUN sudo make install

RUN git clone https://github.com/NVIDIA/nccl.git && cd nccl \
    && make -j8 all && \
    sudo make install

WORKDIR /home/custom

##################### CAFFE #####################

# Set caffe root as working directory 

ENV CAFFE_ROOT=/usr/local/caffe
WORKDIR $CAFFE_ROOT

# Run Caffe installation scripts
ENV CLONE_TAG=master

ENV CMAKE_ARGS=-DUSE_NCCL=ON
RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/BVLC/caffe.git . && \
    for req in $(cat python/requirements.txt) pydot; do pip install $req; done 

RUN cp -f Makefile.config.example Makefile.config && \
    printf "USE_CUDNN := 1\nWITH_PYTHON_LAYER := 1\nUSE_NCCL := 1" >> Makefile.config
RUN make -j"$(nproc)" all && \
    make pycaffe

# Export environment variables manually
#ENV LD_LIBRARY_PATH=/usr/local/caffe/boost_1_61_0_stage_lib:$LD_LIBRARY_PATH \
#    DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH

RUN chmod -R 755 $CAFFE_ROOT

RUN sudo ln /dev/null /dev/raw1394 # Avoid error "libdc1394 error: Failed to initialize libdc1394"

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

WORKDIR /workspace

RUN mkdir -p ~/.ssh
RUN echo "LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH" >> ~/.ssh/environment && \
    echo "PATH=$PATH" >> ~/.ssh/environment

# Copy the files to the necessary folder
COPY toolkit-execute /home/custom

# Make sure the permissions are correct
RUN chmod u+x /home/custom/toolkit-execute
