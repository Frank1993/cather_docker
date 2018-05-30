FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7
# This docker file inherit from base philly-openmpi and install cuda-aware mpi & pytorch & torchvision & nltk & gym
# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker 
LABEL description="This docker has pytorch 0.4.0 torchvision in anaconda python36 with cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/pytorchvision" \
      tag="pytorch-0.4.0-mpi-py36" \
      creator="annelo" tooltype="pytorch" \
      tooltypeversion="0.4.0" \
      createtime="28/5/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

# install openmpi with cuda
RUN cd ~ && \
    wget -q -O - https://www.open-mpi.org/software/ompi/v${OPENMPI_VERSIONBASE}/downloads/${OPENMPI_STRING}.tar.gz | tar -xzf - && \
    cd ${OPENMPI_STRING} && \
    ./configure --prefix=/usr/local/${OPENMPI_STRING} --with-cuda && \
    make -j"$(nproc)" install && \
    rm -rf $STAGE_DIR/${OPENMPI_STRING} && \
    ln -s /usr/local/${OPENMPI_STRING} /usr/local/mpi && \
    # Sanity check:
    test -f /usr/local/mpi/bin/mpic++

RUN apt-get update && apt-get install -y --no-install-recommends \
         git \
         ca-certificates \
         libnccl2 \
         libnccl-dev \
         libjpeg-dev \
         libpng-dev &&\
         rm -rf /var/lib/apt/lists/*

ENV PYTHON_VERSION=3.6

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN cd ~ &&\
    conda install -y numpy pyyaml scipy ipython mkl && \
    conda install -y cuda90 pytorch=0.4.0 torchvision nltk -c pytorch && \
    conda install -y opencv matplotlib mkl-include && \
    conda clean -ya && \
    pip install gym[all]

RUN cd ~ &&\
    git clone --recursive https://github.com/pytorch/pytorch && \
    cd pytorch && git checkout v0.4.0 && git submodule update --init && python setup.py install 

WORKDIR /workspace
RUN chmod -R a+w /workspace
