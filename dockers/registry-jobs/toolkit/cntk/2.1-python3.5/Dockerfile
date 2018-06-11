FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.8.0-cudnn.6

# Labels for the docker
LABEL description="This docker has cntk 2.1 and python3.5" \
      repository="philly/jobs/custom/cntk" \
      tag="2.1-python3.5" \
      creator="vivram" \
      tooltype="cntk" \
      tooltypeversion="2.1" \
      createtime="5/1/2018"

WORKDIR /usr/local/

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates

# Install python environment.
ARG MINICONDA_PREFIX=/usr/local/miniconda3
ARG MINICONDA_VERSION=4.3.11
RUN MINICONDA=Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh && \
    wget --no-verbose https://repo.continuum.io/miniconda/$MINICONDA && \
    chmod a+x $MINICONDA && \
    ./$MINICONDA -b -p $MINICONDA_PREFIX && \
    rm ./$MINICONDA

# Install NCCL.
ARG NCCL_VERSION=2.0.5-2
RUN wget --no-verbose http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl2_${NCCL_VERSION}+cuda8.0_amd64.deb && \
    wget --no-verbose http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl-dev_${NCCL_VERSION}+cuda8.0_amd64.deb && \
    dpkg -i libnccl2_${NCCL_VERSION}+cuda8.0_amd64.deb && \
    dpkg -i libnccl-dev_${NCCL_VERSION}+cuda8.0_amd64.deb && \
    rm libnccl*

# Install CNTK, skipping the openmpi installation and anaconda installation.
RUN wget -q -O - https://cntk.ai/BinaryDrop/CNTK-2-1-Linux-64bit-GPU.tar.gz | tar -xzf - && \
    cd cntk/Scripts/install/linux && \
    sed -i "s:PACKAGES+=\" openmpi-bin\"::" ./install-cntk.sh  && \
    ./install-cntk.sh --py-version 35 --anaconda-basepath /usr/local/miniconda3 --docker && \
    # In the current setup /etc/environment explicitly overrides the PATH variable. Hence setting it explicitly based on current PATH value.
    echo "PATH=$PATH" >> /etc/environment

COPY toolkit-execute /home/job/toolkit-execute
RUN chmod a+x /home/job/toolkit-execute

