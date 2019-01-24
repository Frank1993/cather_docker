FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 0.4.1 in anaconda python36 with cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/pytorch" \
      tag="v0.4.1-py36-det" \
      creator="v-tich" tooltype="pytorch" \
      tooltypeversion="0.4.1" \
      createtime="24/1/2019"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get update && apt-get install -y --no-install-recommends \
         git \
         curl \
         vim \
         cmake \
         tmux \
         htop \
         ca-certificates \
         libnccl2 \
         libnccl-dev \
         libjpeg-dev \
         libyaml-dev \
         libpng-dev &&\
         rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

ENV PYTHON_VERSION=3.6

RUN curl -o ~/anaconda.sh -O  https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh  && \
     chmod +x ~/anaconda.sh && \
     ~/anaconda.sh -b -p /opt/conda && \
     rm ~/anaconda.sh

ENV PATH /opt/conda/bin/:$PATH

RUN conda install torchvision cuda90 pytorch=0.4.1 -c pytorch && conda clean -ya

RUN pip install --upgrade pip && \
    pip install EasyDict==1.7 \
                opencv-python==3.4.1.15 \
                tensorboardX>=1.2 \
                shapely==1.6.4 \
                Cython \
                scipy \
                pyyaml \
                json_tricks \
                scikit-image \
                mmcv \
                addict \
                tqdm

RUN git clone https://github.com/cocodataset/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && python setup.py build_ext install


WORKDIR /workspace
RUN chmod -R a+w /workspace
