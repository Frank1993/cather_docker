FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has tensorflow 16 using python34, with numpy, scipy" \
      repository="philly/jobs/custom/tensorflow" \
      tag="tf16-py34" \
      creator="v-wenwa" tooltype="tensorflow" \
      tooltypeversion="1.6" \
      createtime="5/10/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get -y update && \
    apt-get -y install software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get -y update && \
    apt-get -y install python3.4

RUN curl https://bootstrap.pypa.io/get-pip.py | sudo python3.4

RUN python3.4 -m pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.6.0-cp34-cp34m-linux_x86_64.whl
RUN python3.4 -m pip install --upgrade numpy scipy
