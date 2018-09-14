FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.8.0-cudnn.6

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch0.4.0 using python35, with numpy, scipy, h5py, opencv, six, termcolor, tqdm, zmq, tabulate, tensorboardX, tensorflow-cpu, tensorboard, torchvision" \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch0.4.0-py35" \
      creator="v-wesu" tooltype="pytorch" \
      tooltypeversion="0.4.0" \
      createtime="9/14/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        vim \
        tmux \
        htop \
        unzip

RUN apt-get update && apt-get -y install python3-pip

RUN python3 -m pip install --upgrade numpy scipy opencv-python h5py six termcolor tqdm zmq tabulate torch==0.4.0 torchvision tensorboardX tensorflow
