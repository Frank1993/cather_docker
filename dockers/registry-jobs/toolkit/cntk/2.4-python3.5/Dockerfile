FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Labels for the docker
LABEL description="This docker has cntk 2.4 and python3.5" \
      repository="philly/jobs/custom/cntk" \
      tag="2.4-python3.5" \
      creator="vivram" \
      tooltype="cntk" \
      tooltypeversion="2.4" \
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

# Install CNTK, skipping the openmpi installation and anaconda installation.
RUN wget -q -O - https://cntk.ai/BinaryDrop/CNTK-2-4-Linux-64bit-GPU.tar.gz | tar -xzf - && \
    cd cntk/Scripts/install/linux && \
    sed -i "s:PACKAGES+=\" openmpi-bin\"::" ./install-cntk.sh  && \
    ./install-cntk.sh --py-version 35 --anaconda-basepath /usr/local/miniconda3 --docker && \
    # In the current setup /etc/environment explicitly overrides the PATH variable. Hence setting it explicitly based on current PATH value.
    echo "PATH=$PATH" >> /etc/environment

COPY toolkit-execute /home/job/toolkit-execute
RUN chmod a+x /home/job/toolkit-execute

