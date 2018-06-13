FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.8.0-cudnn.5

# Copy the files to the necessary folder
RUN mkdir /home/job
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod a+rwx /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has tensorflow 0.12.0 and python3" \
      repository="philly/jobs/custom/tensorflow" \
      tag="tf0120-py3" \
      creator="v-qizhou" tooltype="tensorflow" \
      tooltypeversion="0.12" \
      createtime="6/13/2018"

RUN apt-get -y update && \
    apt-get -y install \
    python3-pip

RUN pip3 install --upgrade tensorflow-gpu==1.2 h5py
