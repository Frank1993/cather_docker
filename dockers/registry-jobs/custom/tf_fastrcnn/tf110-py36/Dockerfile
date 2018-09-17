FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod 777 /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has tensorflow latest using python36, with packages required by tensorflow object detection api" \
      repository="philly/jobs/custom/tf_fastrcnn" \
      tag="tf110-py36" \
      creator="v-minghl" tooltype="tf_fastrcnn" \
      tooltypeversion="1.0" \
      createtime="9/14/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get update
RUN apt-get -y install software-properties-common libcupti-dev

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get -y update && \
    apt-get -y install python3.6 python3.6-dev python3.6-tk 
RUN curl https://bootstrap.pypa.io/get-pip.py | sudo python3.6

RUN python3.6 -m pip install --upgrade numpy scipy h5py tensorflow-gpu cython easydict ipython opencv-python setproctitle msgpack msgpack_numpy zmq tqdm matplotlib contextlib2 pillow lxml jupyter

ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

