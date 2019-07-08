FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:philly-ubuntu.16.04-cuda.10.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 1.1.0 in anaconda python36 with cuda 10.0, cudnn 7.0, and nccl2." \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch1.1.0-py36-cuda10-nccl" \
      creator="v-jinhzh" tooltype="pytorch" \
      tooltypeversion="1.1.0" \
      createtime="07/08/2019"
# Everything above this line is required for the docker.
# Add your personalized features below here.


#RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
#RUN wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb && apt install ./cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
#RUN wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb && apt install ./nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb

RUN apt-get update && apt-get install -y --allow-change-held-packages --no-install-recommends \
         git \
         ca-certificates \
         libnccl2=2.4.2-1+cuda10.0 \
         libjpeg-dev \
         libpng-dev &&\
         rm -rf /var/lib/apt/lists/*

ENV PYTHON_VERSION=3.6
ENV LANG C.UTF-8

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda create -y --name pytorch-py$PYTHON_VERSION python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl tqdm&& \
     /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/envs/pytorch-py$PYTHON_VERSION/bin:$PATH

RUN /opt/conda/bin/conda install --name pytorch-py$PYTHON_VERSION pytorch=1.1.0 torchvision -c pytorch && \
    /opt/conda/bin/conda clean -ya

WORKDIR /workspace
RUN chmod -R a+w /workspace

