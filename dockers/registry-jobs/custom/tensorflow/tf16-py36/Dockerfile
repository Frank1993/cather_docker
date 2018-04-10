FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has tensorflow 1.6 using python36, with numpy, scipy, and h5py" \
      repository="philly/jobs/custom/tensorflow" \
      tag="tf16-py36" \
      creator="annelo" tooltype="tensorflow" \
      tooltypeversion="1.6" \
      createtime="4/02/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get update
RUN apt-get -y install software-properties-common libcupti-dev

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get -y update && \
    apt-get -y install python3.6
RUN curl https://bootstrap.pypa.io/get-pip.py | sudo python3.6

RUN python3.6 -m pip install --upgrade numpy scipy h5py https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.6.0-cp36-cp36m-linux_x86_64.whl

ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH