FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod a+rwx /home/job/toolkit-execute

# Labels for the docker
LABEL   description="This docker contains XGBoost implementation and adds requests for usage with Hyperdrive" \
        repository="philly/jobs/custom/xgboost" \
        tag="py36" \
        creator="joelpf" tooltype="xgboost" \
        tooltypeversion="0.71" \
        createtime="4/20/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.
RUN apt-get update && apt-get install -y apt-transport-https
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        tmux \
        curl \
        libjpeg-dev \
        ca-certificates \
        libpng-dev \
        libglu1-mesa-dev \
        freeglut3-dev \
        mesa-common-dev
#Install and Build XGBoost
# RUN git clone --recursive https://github.com/dmlc/xgboost
# RUN cd xgboost; make -j4
# Install Anaconda for python 3.6
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
      wget --quiet http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
      /bin/bash ~/miniconda.sh -b -p /opt/conda && \
      rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH
RUN conda install -y python=3.6 \
        bcolz \
        h5py \
        matplotlib \
        mkl \
        nose \
        notebook \
        Pillow \
        pandas \
        pygpu \
        pyyaml \
        requests \
        scikit-learn \
        six \
        mkl-service \
        xlrd

RUN pip install --upgrade pip numpy scipy pandas sklearn xgboost
ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
RUN env | grep _ >> /etc/environment
