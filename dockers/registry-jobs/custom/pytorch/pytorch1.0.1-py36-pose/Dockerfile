FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 1.0.1 in anaconda python36 with cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/pytorch" \
      tag="v1.0.1-py36-vcpose" \
      creator="bixi" tooltype="pytorch" \
      tooltypeversion="1.0.0" \
      createtime="18/02/2019"
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
        libgl1-mesa-glx \
         libpng-dev &&\
         rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

ENV PYTHON_VERSION=3.6

RUN curl -o ~/anaconda.sh -O  https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh  && \
     chmod +x ~/anaconda.sh && \
     ~/anaconda.sh -b -p /opt/conda && \
     rm ~/anaconda.sh

ENV PATH /opt/conda/bin/:$PATH

RUN conda install torchvision cudatoolkit=9.0 pytorch=1.0.1 -c pytorch && conda clean -ya

RUN pip install --upgrade pip && \
    pip install EasyDict==1.7 \
                opencv-python==3.4.1.15 \
                tensorboardX>=1.2 \
                shapely==1.6.4 \
                yacs==0.1.5 \
                Cython \
                scipy \
                pyyaml \
                json_tricks \
                scikit-image \
                pandas \
                munkres \
                tqdm

# Install Horovod
RUN HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITHOUT_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod==0.15.2 && \
    ldconfig

# Install apex
WORKDIR $STAGE_DIR
RUN pip uninstall -y apex || :
# SHA is something the user can touch to force recreation of this Docker layer, 
# and therefore force cloning of the latest version of Apex
RUN SHA=ToUcHMe git clone https://github.com/NVIDIA/apex.git
WORKDIR $STAGE_DIR/apex
RUN python setup.py install --cuda_ext --cpp_ext
WORKDIR $STAGE_DIR
RUN rm -rf apex

RUN cd / && git clone https://github.com/cocodataset/cocoapi.git && cd cocoapi/PythonAPI && python setup.py build_ext install

WORKDIR /workspace
RUN chmod -R a+w /workspace

