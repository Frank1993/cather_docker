FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:philly-ubuntu.16.04-cuda.10.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 1.1.0 in anaconda python36 with cuda 10.0, cudnn 7.0, and nccl2." \
      repository="philly/jobs/custom/pytorch" \
      tag="py36_torch11_cuda10_matting" \
      creator="v-zhli8" tooltype="pytorch" \
      tooltypeversion="1.1.0" \
      createtime="07/04/2019"
# Everything above this line is required for the docker.
# Add your personalized features below here.


RUN apt-get update && apt-get install -y --allow-change-held-packages --no-install-recommends \
         git \
         ca-certificates locales \
         libnccl2=2.4.2-1+cuda10.0 \
         libjpeg-dev \
         libpng-dev &&\
         locale-gen en_US.UTF-8 && \
         rm -rf /var/lib/apt/lists/*

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
RUN locale

ENV PYTHON_VERSION=3.6
ENV PATH /usr/local/nvidia/bin:/usr/local/nvidia/lib64:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN sudo apt-get update && sudo apt-get install -y vim tmux
RUN echo "alias ll='ls -alF'" >> ~/.bashrc && \
    echo "alias la='ls -A'" >> ~/.bashrc && \
    echo "alias l='ls -CF'" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    mercurial subversion

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    /opt/conda/bin/conda create -y --name py36_torch11 python=$PYTHON_VERSION numpy pyyaml h5py scipy ipython mkl tqdm&& \
    /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/envs/py36_torch11/bin:$PATH

RUN /opt/conda/bin/conda install --name py36_torch11 pytorch=1.1.0 torchvision -c pytorch && \
    /opt/conda/bin/conda install --name py36_torch11 -c conda-forge librosa && \
    /opt/conda/bin/conda clean -ya

RUN pip install opencv-python

WORKDIR /workspace
RUN chmod -R a+w /workspace

