FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7



# Copy the files to the necessary folder

COPY toolkit-execute /home/job/toolkit-execute

RUN chmod u+x /home/job/toolkit-execute



# Labels for the docker

LABEL description="This docker has custom pytorch in python27 for project fiddle with cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/pytorch-elastic" \
      tag="0.4.0" \
      creator="t-suc" tooltype="pytorch" \
      tooltypeversion="0.4.0" \
      createtime="7/24/2018"



COPY nccl-repo-ubuntu1604-2.1.15-ga-cuda9.0_1-1_amd64.deb /home/job

RUN dpkg -i /home/job/nccl-repo-ubuntu1604-2.1.15-ga-cuda9.0_1-1_amd64.deb

RUN apt-get update && apt-get install -y --no-install-recommends \
         git \
         unzip \
         libnccl2=2.1.15-1+cuda9.0 \
         libnccl-dev=2.1.15-1+cuda9.0 \
         ca-certificates \
         libjpeg-dev \
         libpng-dev &&\
         rm -rf /var/lib/apt/lists/*

# Get Conda-ified Python.

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

# Install general libraries

RUN conda install -y python=2.7 numpy pyyaml scipy ipython mkl scikit-learn matplotlib pandas setuptools Cython h5py

RUN conda clean -ya

RUN conda install mkl-include cmake cffi typing

RUN conda install -c mingfeima mkldnn


# pip packages

COPY requirements.txt /home/job

RUN /opt/conda/bin/pip install -r /home/job/requirements.txt


# Make ssh dir
RUN mkdir /root/.ssh/

# Copy over private key, and set permissions
ADD id_rsa /root/.ssh/id_rsa
ADD id_rsa.pub /root/.ssh/id_rsa.pub
RUN chmod 400  /root/.ssh/id_rsa


# Create known_hosts
RUN touch /root/.ssh/known_hosts

# Add bitbucket key
RUN ssh-keyscan bitbucket.org >> /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts


# ...get PyTorch source and build.

RUN rm -rf pytorch

RUN git clone -b elastic --recursive git@github.com:deepakn94/pytorch_private.git pytorch

RUN cd pytorch ; python setup.py install; cd ..


# Set CUDA_ROOT

RUN export CUDA_HOME="/usr/local/cuda" 