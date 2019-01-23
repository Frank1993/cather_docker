FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.8.0-cudnn.6

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has tensorflow 1.30 using python3.5, with numpy nltk tqdm pandas scikit-learn scipy ujson h5py" \
      repository="philly/jobs/custom/tensorflow" \
      tag="tf130-py35-nltk" \
      creator="v-haizhu" tooltype="tensorflow" \
      tooltypeversion="1.30" \
      createtime="1/22/2019"
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

RUN python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.3.0-cp35-cp35m-linux_x86_64.whl

RUN python3 -m pip install nltk==3.2.4
RUN python3 -m pip install numpy==1.13.1
RUN python3 -m pip install pandas==0.20.1
RUN python3 -m pip install scikit-learn==0.18.1
RUN python3 -m pip install scipy==0.19.0
RUN python3 -m pip install tqdm==4.14.0
RUN python3 -m pip install ujson==1.35
RUN python3 -m pip install --upgrade h5py

ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
