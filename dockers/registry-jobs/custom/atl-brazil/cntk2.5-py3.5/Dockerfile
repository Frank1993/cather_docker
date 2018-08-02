FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
RUN mkdir /home/job
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has cntk 2.5 using anaconda python35 with other toolkits required by the ATL-Brazil team" \
repository="philly/jobs/custom/atl-brazil" \
tag="cntk2.5-py3.5" \
creator="v-rodemo" tooltype="cntk" \
tooltypeversion="2.5" \
createtime="3/23/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update -y && \
apt-get install -y --no-install-recommends \
wget \
build-essential \
unzip \
xvfb \
libsndfile-dev \
git \
cmake \
libhdf5-dev \
libjasper-dev \
libjpeg-dev \
libpng-dev \
libtiff-dev && \
rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Installing Miniconda
RUN echo 'export PATH=/opt/conda/envs/cntk-py35/bin:$PATH' > /etc/profile.d/conda.sh && \
wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.4.10-Linux-x86_64.sh -O ~/anaconda.sh && \
/bin/bash ~/anaconda.sh -b -p /opt/conda && \
rm ~/anaconda.sh

## Error in opencv, updating conda
RUN /opt/conda/bin/conda update --yes -n base conda

## Adding cntk-py35 to the environment

ENV PATH /opt/conda/envs/cntk-py35/bin:$PATH

## Creating the environment
RUN /opt/conda/bin/conda create -n cntk-py35 --yes python=3.5 pandas h5py jsonschema numpy scipy scikit-learn pillow pytables pyyaml && \
    pip install easydict azure && \
    /opt/conda/bin/conda install --yes --name cntk-py35 opencv=3.1 -c conda-forge && \
    /opt/conda/bin/conda clean --yes --all

ENV LD_LIBRARY_PATH /opt/conda/envs/cntk-py35/lib/:$LD_LIBRARY_PATH

# CNTK configuration

## MKL
RUN mkdir /usr/local/mklml
RUN wget https://github.com/01org/mkl-dnn/releases/download/v0.12/mklml_lnx_2018.0.1.20171227.tgz && \
tar -xzf mklml_lnx_2018.0.1.20171227.tgz -C /usr/local/mklml && \
wget --no-verbose -O - https://github.com/01org/mkl-dnn/archive/v0.12.tar.gz | tar -xzf - && \
cd mkl-dnn-0.12 && \
ln -s /usr/local external && \
mkdir -p build && \
cd build && \
cmake .. && \
make && \
make install && \
cd ../.. && \
rm -rf mkl-dnn-0.12
ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH

## CNTK
RUN pip install cntk-gpu