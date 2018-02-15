FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has latest github version of cupy and chainer using anaconda python36 with other toolkits" \
      repository="philly/jobs/custom/chainer" \
      tag="conda_py36_chainer_cuda9" \
      creator="haerdoga" tooltype="chainer" \
      tooltypeversion="1.0" \
      createtime="2/12/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    wget \
    xvfb \
    libsndfile-dev \
    git \
    libhdf5-dev && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN conda install pandas
RUN conda install h5py
RUN pip install soundfile
RUN pip install librosa
RUN pip install mir_eval
RUN pip install pathos
RUN pip install pymongo
RUN pip install monk

RUN cd /home/ && git clone https://github.com/cupy/cupy.git && cd cupy && python setup.py install
RUN cd /home/ && git clone https://github.com/chainer/chainer.git && cd chainer && python setup.py install
