FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has tensorflow 1.5 using python35" \
      repository="philly/jobs/custom/tf5_py35_cudnn90_libcudnn7415_bilm" \
      tag="tf5_py35_cudnn90_libcudnn7415_bilm" \
      creator="chuah" tooltype="tf5_py35_cudnn90_libcudnn7415_bilm" \
      tooltypeversion="1.5" \
      createtime="12/22/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get update

RUN apt-get install -y zip unzip vim tmux git 

RUN apt-get update

RUN apt-get install -y --no-install-recommends libcudnn7=7.4.1.5-1+cuda9.0

RUN apt-get -y install software-properties-common libcupti-dev

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get -y update && \
    apt-get -y install python3.5 python3.5-dev python3.5-tk
RUN curl https://bootstrap.pypa.io/get-pip.py | sudo python3.5

RUN python3.5 -m pip install --upgrade \
Markdown \
matplotlib \
protobuf \
tensorflow-gpu==1.5 \
tensorflow==1.5 \
tqdm \
h5py

ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN cd /home/job/; git clone https://github.com/allenai/bilm-tf; cd bilm-tf; python3.5 setup.py install
