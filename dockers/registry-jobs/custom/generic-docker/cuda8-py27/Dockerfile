FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.8.0-cudnn.6

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker contains chainer 1.24 on cuda8 with python v2.7 from anaconda" \
      repository="philly/jobs/custom/generic-docker" \
      tag="cuda8-py27" \
      creator="retachet" tooltype="generic-docker" \
      tooltypeversion="1.0" \
      createtime="4/18/2018"
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
# Install Anaconda for python 2.7
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
      wget --quiet http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
      /bin/bash ~/miniconda.sh -b -p /opt/conda && \
      rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH
RUN conda install -y python=2.7 \
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
RUN pip install --no-cache-dir filelock pillow cython cupy-cuda80

ENV CUDA_PATH /usr/local/cuda
ENV CUDA_HOME /usr/local/cuda
ENV PATH /usr/local/cuda/bin:${PATH}
ENV CPATH /usr/include:/usr/local/cuda/bin:$CPATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
ENV CFLAGS -I/usr/local/cuda/include/
ENV LDFLAGS -L/usr/local/cuda/lib64/

RUN cd ~
RUN git clone https://github.com/chainer/chainer.git && cd chainer && git checkout tags/v1.24.0
RUN cd chainer && python setup.py install
RUN env | grep _ >> /etc/environment
RUN sudo chmod -R ugo+rw /opt/conda