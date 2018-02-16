# DockerFile to run executables for Machine-Translation group's training process

FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.8.0-cudnn.6

# --- Labels for the docker
LABEL description="This docker image runs executables for the Machine-Translation group's training process." \
      repository="philly/jobs/custom/generic-commandline-for-philly-submit-tool" \
      tag="v0.1" \
      creator="fseide" tooltype="generic" \
      tooltypeversion="any" \
      createtime="2/5/2018"

WORKDIR /usr/local/

# --- Install Mono environment for SmartFlow
RUN apt-get update && \
    apt -y install mono-mcs && \
    apt -y install libmono-system-runtime-serialization-formatters-soap4.0-cil

# --- Install Boost
RUN apt-get install -y --no-install-recommends \
    libbz2-dev \
    python-dev

ARG BOOST_VERSION=1.60.0
RUN BOOST_UNDERSCORE_VERSION=$(echo $BOOST_VERSION | tr . _) && \
    wget -q -O - https://sourceforge.net/projects/boost/files/boost/${BOOST_VERSION}/boost_${BOOST_UNDERSCORE_VERSION}.tar.gz/download | tar -xzf - && \
    cd boost_${BOOST_UNDERSCORE_VERSION} && \
    ./bootstrap.sh --prefix=/usr/local/boost-${BOOST_VERSION}  && \
    ./b2 -d0 -j $(nproc) install && \
    cd .. && \
    rm -rf boost_${BOOST_UNDERSCORE_VERSION}
ENV LD_LIBRARY_PATH /usr/local/boost-1.60.0:$LD_LIBRARY_PATH

# Install python environment.
# Currently, neither Dynamite nor Marian needs this, so we disable it for now, to keep things snappy.
#ARG MINICONDA_PREFIX=/usr/local/miniconda3
#ARG MINICONDA_VERSION=4.3.11
#RUN MINICONDA=Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh && \
#    wget --no-check-certificate --no-verbose https://repo.continuum.io/miniconda/$MINICONDA && \
#    chmod a+x $MINICONDA && \
#    ./$MINICONDA -b -p $MINICONDA_PREFIX && \
#    rm ./$MINICONDA

# --- Install NCCL.
ARG NCCL_VERSION=2.0.5-2
RUN wget --no-verbose http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl2_${NCCL_VERSION}+cuda8.0_amd64.deb && \
    wget --no-verbose http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl-dev_${NCCL_VERSION}+cuda8.0_amd64.deb && \
    dpkg -i libnccl2_${NCCL_VERSION}+cuda8.0_amd64.deb && \
    dpkg -i libnccl-dev_${NCCL_VERSION}+cuda8.0_amd64.deb && \
    rm libnccl*

WORKDIR /root/
COPY toolkit-execute /home/job/toolkit-execute
