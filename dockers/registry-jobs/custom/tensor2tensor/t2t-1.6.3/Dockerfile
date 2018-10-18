FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has tensor2tensor, based on the original tf16-py36 image" \
      repository="philly/jobs/custom/tensor2tensor" \
      tag="t2t-1.6.3" \
      creator="tage" tooltype="tensor2tensor" \
      tooltypeversion="1.6.3" \
      createtime="6/14/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.
RUN apt-get -y update
RUN apt-get -y install software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get -y update && \
    apt-get -y install python3.6
RUN curl https://bootstrap.pypa.io/get-pip.py | sudo python3.6

RUN python3.6 -m pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.6.0-cp36-cp36m-linux_x86_64.whl numpy scipy opencv-python h5py tensor2tensor oauth2client six
