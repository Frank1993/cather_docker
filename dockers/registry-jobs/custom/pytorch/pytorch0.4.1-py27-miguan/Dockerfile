FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 0.4.1 in anaconda python27 with cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/pytorch" \ 
      tag="pytorch0.4.1-py27-miguan" \
      creator="v-miguan" tooltype="pytorch" \
      tooltypeversion="0.4.1" \
      createtime="04/16/2019"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get update
RUN apt-get -y install python-pip
RUN apt-get -y install git
RUN apt-get -y install vim
RUN apt-get -y install htop
RUN apt-get -y install tmux
RUN apt-get -y install unzip
RUN apt-get -y install zip
RUN apt-get -y install libsndfile-dev

RUN pip install --upgrade pip
RUN python -m pip install --upgrade torch==0.4.1 torchvision tensorboardX
RUN python -m pip install --upgrade dominate requests 
RUN python -m pip install --upgrade opencv-python h5py six termcolor zmq tabulate
RUN python -m pip install --upgrade matplotlib==2.2.3 numpy==1.16.1 scipy==1.2.1 tqdm==4.31.1

RUN apt-get -y install libnccl2=2.3.7-1+cuda9.0 libnccl-dev=2.3.7-1+cuda9.0

RUN HOROVOD_GPU_ALLREDUCE=NCCL python -m pip install --no-cache-dir horovod

