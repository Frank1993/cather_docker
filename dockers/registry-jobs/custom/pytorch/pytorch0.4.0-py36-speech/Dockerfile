FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 0.4.0 in python36 with cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch0.4.0-py36-speech" \
      creator="v-wanke" tooltype="pytorch" \
      tooltypeversion="0.4.0" \
      createtime="05/04/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get update && apt-get install -y --no-install-recommends \
         git \
         python3-pip \
         octave &&\
         rm -rf /var/lib/apt/lists/*

ENV PYTHON_VERSION=3.6

ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
RUN pip3 install --upgrade pip
RUN pip3 install setuptools numpy scipy scikit-learn
RUN pip3 install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp35-cp35m-linux_x86_64.whl
RUN pip3 install torchvision

WORKDIR /workspace
RUN chmod -R a+w /workspace
