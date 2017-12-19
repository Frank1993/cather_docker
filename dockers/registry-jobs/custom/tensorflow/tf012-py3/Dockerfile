FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.8.0-cudnn.5

# Copy the files to the necessary folder
RUN mkdir /home/job
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod a+rwx /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has tensorflow 0.12 and python3" \
      repository="philly/jobs/custom/tensorflow" \
      tag="tf012-py3" \
      creator="viye" tooltype="tensorflow" \
      tooltypeversion="0.12" \
      createtime="12/18/2017"

RUN apt-get -y update && \
    apt-get -y install \
    python3-pip

ENV TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp35-cp35m-linux_x86_64.whl

RUN pip3 install --upgrade $TF_BINARY_URL
