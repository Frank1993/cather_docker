FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:philly-ubuntu.16.04-cuda.10.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has tensorflow 1.13 with NCCL2, includes: h5py, bz2file, in anaconda python36 with cuda 9.0, and cudnn 7.0." \
      lepository="philly/jobs/custom/tensorflow" \
      tag="tf113-py36-nccl2" \
      creator="v-hanbao" tooltype="tensorflow" \
      tooltypeversion="1.13.1" \
      createtime="03/06/2019"
# Everything above this line is required for the docker.
# Add your personalized features below here.

# RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
# RUN wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.0.130-1_amd64.deb && apt install ./cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
# RUN wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb && apt install ./nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb

RUN apt-get update && apt-get install -y --allow-change-held-packages --no-install-recommends \
         git \
         ca-certificates \
         libnccl2=2.4.2-1+cuda10.0 \
         libjpeg-dev \
         libpng-dev &&\
         rm -rf /var/lib/apt/lists/*


ENV PYTHON_VERSION=3.6
ENV TENSORFLOW_VERSION=1.13.1

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda create -y --name tensorflow-py$PYTHON_VERSION python=$PYTHON_VERSION numpy h5py bz2file pyyaml scipy ipython mkl && \
     /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/envs/tensorflow-py$PYTHON_VERSION/bin:$PATH


RUN pip install tensorflow-gpu==${TENSORFLOW_VERSION} h5py tqdm

WORKDIR /workspace
RUN chmod -R a+w /workspace

