FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.8.0-cudnn.6

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has tensorflow 0.12.1 using python3.5, with numpy nltk tqdm pandas scikit-learn scipy ujson h5py" \
      repository="philly/jobs/custom/tensorflow" \
      tag="tf0121-py3-bna" \
      creator="v-haizhu" tooltype="tensorflow" \
      tooltypeversion="0.12.1" \
      createtime="1/23/2019"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        vim \
        tmux \
        htop

RUN apt-get update && apt-get -y install python3-pip

RUN python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp35-cp35m-linux_x86_64.whl

RUN python3 -m pip install nltk==3.2.1

RUN python3 -m pip install --upgrade numpy pandas scikit-learn scipy tqdm ujson h5py

ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
