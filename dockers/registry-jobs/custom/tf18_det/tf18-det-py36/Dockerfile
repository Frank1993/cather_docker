FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has tensorflow 1.8 using python36, with numpy, scipy, cython, easydict, ipython, opencv-python, setproctitle, msgpack, msgpack_numpy, zmq, tqdm, matplotlib, pyyaml, and h5py" \
      repository="philly/jobs/custom/tf18_det" \
      tag="tf18-det-py36" \
      creator="hanhu" tooltype="tf18_det" \
      tooltypeversion="1.8" \
      createtime="5/27/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get update

RUN apt-get install -y --no-install-recommends libcudnn7=7.0.5.15-1+cuda9.0

RUN apt-get -y install software-properties-common libcupti-dev

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get -y update && \
    apt-get -y install python3.6 python3.6-dev python3.6-tk
RUN curl https://bootstrap.pypa.io/get-pip.py | sudo python3.6

RUN python3.6 -m pip install --upgrade numpy scipy h5py https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.8.0-cp36-cp36m-linux_x86_64.whl cython easydict ipython opencv-python setproctitle msgpack msgpack_numpy zmq tqdm matplotlib pyyaml

ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

