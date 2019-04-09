FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.8.0-cudnn.6

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has tensor2tensor" \
      repository="philly/jobs/custom/tensor2tensor" \
      tag="t2t-1.2.9" \
      creator="Shufang Xie" tooltype="tensor2tensor" \
      tooltypeversion="1.2.9" \
      createtime="4/9/2019"
# Everything above this line is required for the docker.
# Add your personalized features below here.
RUN apt-get -y update
RUN apt-get -y install software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get -y update && \
    apt-get -y install python3.6
RUN curl https://bootstrap.pypa.io/get-pip.py | sudo python3.6

RUN python3.6 -m pip install --upgrade tensorflow-gpu==1.4.1 numpy scipy opencv-python h5py tensor2tensor==1.2.9
