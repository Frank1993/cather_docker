FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has tensorflow 1.8 using python35" \
      repository="philly/jobs/custom/tf18_cuda90_py35" \
      tag="tf18_cuda90_py35" \
      creator="wei" tooltype="tf18_cuda90_py35" \
      tooltypeversion="1.8" \
      createtime="7/17/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get update

RUN apt-get install -y --no-install-recommends libcudnn7=7.0.5.15-1+cuda9.0

RUN apt-get -y install software-properties-common libcupti-dev

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get -y update && \
    apt-get -y install python3.5 python3.5-dev python3.5-tk
RUN curl https://bootstrap.pypa.io/get-pip.py | sudo python3.5

RUN python3.5 -m pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.8.0-cp35-cp35m-linux_x86_64.whl imageio==2.3.0 opencv-python tqdm better-exceptions==0.2.1 sacred==0.7.3

ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

