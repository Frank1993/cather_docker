FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has tensorflow 1.12 with support for tensor2tensor, in anaconda python36 with cuda 9.0, and cudnn 7.0." \
      lepository="philly/jobs/custom/tensorflow" \
      tag="tf112-py36-t2t" \
      creator="v-yirwa" tooltype="tensorflow" \
      tooltypeversion="1.12.0" \
      createtime="02/18/2019"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
RUN wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb && apt install ./cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
RUN wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb && apt install ./nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb

RUN apt-get update && apt-get install -y --no-install-recommends \
         software-properties-common \
         zip \
         unzip \
         vim \
         octave \
         dos2unix \
         less \
         git \
         ca-certificates \
         libnccl2=2.2.13-1+cuda9.0 \
         libjpeg-dev \
         libpng-dev &&\
         rm -rf /var/lib/apt/lists/*

ENV PYTHON_VERSION=3.6
ENV TENSORFLOW_VERSION=1.12

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda create -y --name tensorflow-py$PYTHON_VERSION python=$PYTHON_VERSION numpy h5py bz2file pyyaml scipy ipython mkl && \
     /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/envs/tensorflow-py$PYTHON_VERSION/bin:$PATH

RUN pip install --upgrade pip
RUN pip install tensorflow-gpu==${TENSORFLOW_VERSION} h5py tqdm
RUN pip install tensorflow-hub==0.1.1
RUN pip install bz2file dopamine-rl flask future gevent
RUN pip install google-api-python-client gunicorn gym mesh-tensorflow oauth2client
RUN pip install opencv-python requests scipy six sympy
RUN pip install tensorflow-probability tf-agents tfds-nightly
RUN pip install absl-py pytest mock pylint jupyter gsutil matplotlib
RUN pip install Pillow==5.1.0
RUN pip install pandas==0.23.0

WORKDIR /workspace
RUN chmod -R a+w /workspace

