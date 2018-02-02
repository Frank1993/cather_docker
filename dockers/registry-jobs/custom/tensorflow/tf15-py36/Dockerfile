FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has tensorflow 15 using python36, with numpy, scipy, h5py, and opencv" \
      repository="philly/jobs/custom/tensorflow" \
      tag="tf15-py36" \
      creator="neel" tooltype="tensorflow" \
      tooltypeversion="1.5" \
      createtime="1/20/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get -y install software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get -y update && \
    apt-get -y install python3.6
RUN curl https://bootstrap.pypa.io/get-pip.py | sudo python3.6

RUN python3.6 -m pip install --upgrade tensorflow-gpu numpy scipy opencv-python h5py
