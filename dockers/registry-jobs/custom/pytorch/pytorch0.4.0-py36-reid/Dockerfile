FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 0.4.0 in python36 with cuda 9.0, and cudnn 7.0 mainly for speech synthesis" \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch0.4.0-py36-reid" \
      creator="v-xuj" tooltype="pytorch" \
      tooltypeversion="0.4.0" \
      createtime="14/8/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.
ENV PYTHON_VERSION=3.6

RUN apt-get update && apt-get install -y --no-install-recommends \
		 python3-pip \
		 libsndfile1 \
		 libsndfile1-dev \
             unzip \
		 git &&\
         rm -rf /var/lib/apt/lists/*
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
RUN pip3 install --upgrade pip
RUN pip3 install setuptools
RUN pip3 install numpy
RUN pip3 install h5py
RUN pip3 install scipy
RUN pip3 install scikit-learn
RUN pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.0-cp35-cp35m-linux_x86_64.whl
RUN pip3 install torchvision
RUN pip3 install python-dateutil
RUN pip3 install six
RUN pip3 install librosa
RUN pip3 install tensorboardX

WORKDIR /workspace
RUN chmod -R a+w /workspace
