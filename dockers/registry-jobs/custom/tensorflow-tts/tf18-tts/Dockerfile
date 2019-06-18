FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has tensorflow 1.8 and all packages Tacotron depends" \
      repository="philly/jobs/custom/tensorflow-tts" \
      tag="tf18-tts" \
      creator="v-haogu" tooltype="tensorflow-tts" \
      tooltypeversion="1.8" \
      createtime="5/26/2018"
# Everything above this line is required for the docker.

# Add your personalized features below here.
RUN apt-get -y update
RUN apt-get -y install git
RUN apt-get -y install htop
RUN apt-get -y install tmux
RUN apt-get -y install unzip
RUN apt-get -y install zip
RUN apt-get -y install libsndfile-dev
RUN apt-get -y install apt-transport-https vim iproute2 net-tools build-essential ca-certificates curl wget 
RUN apt-get -y install software-properties-common
RUN add-apt-repository ppa:jonathonf/python-3.6
RUN apt-get -y update
RUN apt-get -y install python3.6
RUN apt-get -y install python3-pip

RUN python3 -m pip install tensorflow-gpu==1.8.0
RUN python3 -m pip install --upgrade torch==1.1.0 torchvision tensorboardX soundfile jupyter
RUN python3 -m pip install --upgrade opencv-python h5py six termcolor zmq tabulate
RUN python3 -m pip install --upgrade falcon==1.2.0 inflect==0.2.5 librosa==0.5.1 matplotlib==2.0.2 numpy==1.14.3 scipy==0.19.0 tqdm==4.11.2 Unidecode==0.4.20
RUN python3 -m pip install nnmnkwii tensorboard_logger nltk docopt

RUN apt-get -y install libnccl2=2.3.7-1+cuda9.0 libnccl-dev=2.3.7-1+cuda9.0
RUN HOROVOD_GPU_ALLREDUCE=NCCL python3 -m pip install --no-cache-dir horovod
RUN git clone https://github.com/NVIDIA/apex && cd apex && python3 -m pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .