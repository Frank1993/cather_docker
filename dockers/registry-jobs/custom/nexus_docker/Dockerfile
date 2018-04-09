FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has nexus with cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/nexus" \
      tag="nexus" \
      creator="v-yucj" tooltype="nexus" \
      tooltypeversion="1.0" \
      createtime="4/6/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

ENV LD_LIBRARY_PATH /usr/local/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y \
  wget \
  curl \
  zip \
  unzip \
  zlib1g-dev \
  autoconf \
  automake \
  git \
  make \
  cmake \
  clang \
  swig \
  net-tools \
  build-essential \
  libgtk2.0-dev \
  pkg-config \
  mlocate \
  libavcodec-dev \
  libavformat-dev \
  libswscale-dev \
  libboost-all-dev \
  libgflags-dev \
  libgoogle-glog-dev \
  libgtest-dev \
  libopenblas-dev \
  libsnappy-dev \
  libleveldb-dev \
  liblmdb-dev \
  libhdf5-serial-dev \
  libyaml-cpp-dev \
  python-dev \
  python-pip \
  python-setuptools \
  openjdk-8-jdk \
  git

RUN pip install --upgrade pip && pip install numpy scipy

# Install bazel 0.11.1
RUN mkdir -p /tmp && cd /tmp && \
    wget https://github.com/bazelbuild/bazel/releases/download/0.11.1/bazel-0.11.1-without-jdk-installer-linux-x86_64.sh && \
    chmod +x bazel-0.11.1-without-jdk-installer-linux-x86_64.sh && \
    ./bazel-0.11.1-without-jdk-installer-linux-x86_64.sh

# Install opencv 3.1.0
RUN wget https://github.com/Itseez/opencv/archive/3.1.0.zip -P /tmp/ && \
    cd /tmp && unzip 3.1.0.zip && cd opencv-3.1.0 && mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE -D WITH_TIFF=ON -D WITH_IPP=ON \
          -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_C_EXAMPLES=ON \
          -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D WITH_OPENGL=ON \
          -D ENABLE_FAST_MATH=1 -D WITH_CUDA=OFF .. && \
    make -j$(nproc) && make install
# Add OpenCV 3rd party library path
ENV LIBRARY_PATH $LIBRARY_PATH:/usr/local/share/OpenCV/3rdparty/lib/
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/share/OpenCV/3rdparty/lib/

# Install protobuf 3.5.0
RUN cd /tmp && mkdir protobuf && cd protobuf && \
    wget https://github.com/google/protobuf/releases/download/v3.5.0/protoc-3.5.0-linux-x86_64.zip && \
    unzip protoc-3.5.0-linux-x86_64.zip && \
    cp bin/protoc /usr/local/bin/ && \
    wget https://github.com/google/protobuf/releases/download/v3.5.0/protobuf-python-3.5.0.zip && \
    unzip protobuf-python-3.5.0.zip && cd protobuf-3.5.0 && ./configure && \
    make -j$(nproc) && make install

# Install grpc 1.10.x
RUN cd /tmp && git clone -b v1.10.x https://github.com/grpc/grpc && \
    cd grpc && git submodule update --init && make -j$(nproc) HAS_SYSTEM_PROTOBUF=false && \
    make install

RUN rm -r /tmp/*

# Link cudnn to /usr/local/cuda
RUN ln -s /usr/include/cudnn.h /usr/local/cuda/include/
RUN ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so /usr/local/cuda/lib64/
RUN ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.7 /usr/local/cuda/lib64/

# Update a database for mlocate
RUN updatedb

RUN cd / && git clone https://github.com/uwsaml/nexus.git
RUN cd /nexus && git submodule update --init --recursive && make -j$($(nproc)-1) all
ENV PATH /nexus/build/bin:$PATH
