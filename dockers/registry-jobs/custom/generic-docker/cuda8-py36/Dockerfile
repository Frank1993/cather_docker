FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.8.0-cudnn.6

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker contains chainer on cuda8 with python v3.6 from anaconda" \
      repository="philly/jobs/custom/generic-docker" \
      tag="cuda8-py36" \
      creator="retachet" tooltype="generic-docker" \
      tooltypeversion="1.0" \
      createtime="4/24/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      git \
      wget \
      unzip \
      tmux \
      curl \
      libjpeg-dev \
      ca-certificates \
      libpng-dev \
      libglu1-mesa-dev \
      freeglut3-dev \
      mesa-common-dev
# Install Anaconda for python 3.6
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
      wget --quiet http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
      /bin/bash ~/miniconda.sh -b -p /opt/conda && \
      rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH
RUN conda install -y python=3.6 \
      bcolz \
      h5py \
      matplotlib \
      mkl \
      nose \
      notebook \
      Pillow \
      pandas \
      pygpu \
      pyyaml \
      scikit-learn \
      six \
      mkl-service
RUN conda install -c conda-forge tensorboardx
RUN pip install --upgrade numpy scipy nltk click tqdm tables pyprind joblib gym mako cached_property pygame cloudpickle
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_HOME /usr/local/cuda
ENV PATH ${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH ${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
RUN pip install --no-cache-dir cupy-cuda80 chainer==5.0.0a1
RUN env | grep _ >> /etc/environment
RUN sudo chmod -R ugo+rw /opt/conda