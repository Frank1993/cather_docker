FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:philly-ubuntu.16.04-cuda.10.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 0.4.0 in python37 with cuda 10.0, and cudnn 7.0." \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch-stca-speech-latest" \
      creator="wake" tooltype="pytorch" \
      tooltypeversion="1.1.0" \
      createtime="06/08/2019"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get update && apt-get install -y --no-install-recommends \
         git \
         ca-certificates \
         curl \
         vim \
         zip \
         unzip \
         locales \
         language-pack-en \
         libsndfile1 \
         htop && \
     rm -rf /var/lib/apt/lists/*

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
RUN locale-gen en_US.UTF-8 && dpkg-reconfigure locales && locale

ENV PYTHON_VERSION 3.7
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy scipy scikit-learn matplotlib pyyaml tqdm && \
     /opt/conda/bin/conda install -y torchvision pytorch=1.1.0 cudatoolkit=10.0 -c pytorch && \
     /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/usr/local/cuda/include:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/opt/OpenBLAS/lib:$LD_LIBRARY_PATH

RUN pip install --upgrade pip && \
    pip install soundfile librosa pysptk museval tensorboardX

WORKDIR /workspace
RUN chmod -R a+w /workspace
