FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:philly-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 1.1.0 in python37 with cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch-stca-speech-latest" \
      creator="wake" tooltype="pytorch" \
      tooltypeversion="1.1.0" \
      createtime="06/08/2019"
# Everything above this line is required for the docker.
# Add your personalized features below here.

ENV CUDA_VERSION=9.0 \
    PYTHON_VERSION=3.7 \
    PYTORCH_VERSION=1.1.0

RUN apt update && apt install -y --no-install-recommends \
         libnccl2=2.4.2-1+cuda$CUDA_VERSION \
         libnccl-dev=2.4.2-1+cuda$CUDA_VERSION \
         git \
         ca-certificates \
         vim \
         zip \
         unzip \
         libsndfile1 \
         htop && \
    rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy scipy scikit-learn matplotlib pyyaml tqdm && \
     /opt/conda/bin/conda install -y torchvision pytorch=$PYTORCH_VERSION cudatoolkit=$CUDA_VERSION -c pytorch && \
     /opt/conda/bin/conda clean -ya

ENV PATH=/opt/conda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/include:$LD_LIBRARY_PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/opt/OpenBLAS/lib:$LD_LIBRARY_PATH

RUN pip install --upgrade pip && \
    pip install soundfile librosa pysptk museval tensorboardX seaborn

WORKDIR /workspace
RUN chmod -R a+w /workspace
