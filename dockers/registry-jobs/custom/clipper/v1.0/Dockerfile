FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has clipper with cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/clipper" \
      tag="clipper" \
      creator="v-yucj" tooltype="clipper" \
      tooltypeversion="1.0" \
      createtime="5/24/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

ENV LD_LIBRARY_PATH /usr/local/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y g++ automake wget autoconf autoconf-archive libtool libboost-all-dev \
    libevent-dev libdouble-conversion-dev libgoogle-glog-dev libgflags-dev liblz4-dev \
    liblzma-dev libsnappy-dev make zlib1g-dev binutils-dev libjemalloc-dev libssl-dev \
    pkg-config libiberty-dev git cmake libev-dev libhiredis-dev libzmq5 libzmq5-dev build-essential

## Install Folly
RUN git clone https://github.com/facebook/folly \
    && cd folly/folly \
    && git checkout tags/v2017.08.14.00 \
    && autoreconf -ivf \
    && ./configure \
    && make -j4 \
    && make install

RUN cd / && git clone https://github.com/YuchenJin/clipper_caffe2.git

RUN cd / && git clone https://github.com/ucbrise/clipper.git

RUN cd /clipper && git submodule update --init --recursive

RUN cd /clipper/src/libs/spdlog \
    && git apply ../patches/make_spdlog_compile_linux.patch \
    && cd /clipper \
    && ./configure --cleanup-quiet \
    && ./configure --release \
    && cd release \
    && make -j8 management_frontend \
    && make -j8 query_frontend

# Install Redis.
RUN \
  cd /tmp && \
  wget http://download.redis.io/redis-stable.tar.gz && \
  tar xvzf redis-stable.tar.gz && \
  cd redis-stable && \
  make && \
  make install && \
  cp -f src/redis-sentinel /usr/local/bin && \
  mkdir -p /etc/redis && \
  cp -f *.conf /etc/redis && \
  rm -rf /tmp/redis-stable* && \
  sed -i 's/^\(bind .*\)$/# \1/' /etc/redis/redis.conf && \
  sed -i 's/^\(daemonize .*\)$/# \1/' /etc/redis/redis.conf && \
  sed -i 's/^\(dir .*\)$/# \1\ndir \/data/' /etc/redis/redis.conf && \
  sed -i 's/^\(logfile .*\)$/# \1/' /etc/redis/redis.conf

# Define mountable directories.
VOLUME ["/data"]

# Define working directory.
WORKDIR /data

# Define default command.
CMD ["redis-server", "/etc/redis/redis.conf"]

# Expose ports.
EXPOSE 6379
EXPOSE 1338
EXPOSE 1337
EXPOSE 7000