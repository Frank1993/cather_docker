FROM phillyregistry.azurecr.io/philly/jobs/custom/tensorflow:tf13-py36

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="GYM and ALE for Python 3.6 and TensorFlow 1.3 with Keras too" \
      repository="philly/jobs/custom/openai-gym-ale" \
      tag="tf13-py36" \
      creator="retachet" \
      toolkit="tensorflow" \
      tooltypeversion="1.3"

# Everything above this line is required for the docker.
# Add your personalized features below here.

# Handle additional python dependencies
RUN pip --no-cache-dir install \
    jupyter      \
    tqdm         \
    appdirs      \
    bidict       \
    bz2file      \
    gensim       \
    nvidia-ml-py \
    smart-open   \
    pygame       \
    pyyaml       \
    argparse     \
    pandas       \
    imageio

# Install Cython
RUN pip install Cython

RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
  build-essential \
  gfortran \
  git \
  curl \
  wget \
  unzip \
  rsync \
  bzip2 \
  pkg-config \
  ca-certificates \
  libfreetype6-dev \
  libpng12-dev \
  libzmq3-dev \
  liblapack-dev \
  libopenblas-dev \
  libglib2.0-0 libxext6 libsm6 libxrender1 \
  software-properties-common \
  libgtk2.0-0 \
  libhdf5-serial-dev

# Add all environment variables available to the user
RUN env | grep _ >> /etc/environment

# TensorBoard
EXPOSE 6006

# Install opencv
RUN conda install -c https://conda.anaconda.org/menpo opencv3

# Install ALE
RUN conda install libgcc
RUN conda install -c rdonnelly gcc_linux-64 gxx_linux-64

RUN git clone https://github.com/mgbellemare/Arcade-Learning-Environment.git
RUN apt-get install -y --no-install-recommends \
  libsdl1.2-dev \
  libsdl-gfx1.2-dev \
  libsdl-image1.2-dev \
  cmake
RUN cd Arcade-Learning-Environment && mkdir build && cd build && \
  cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON .. && \
  make -j 4 && cd .. && \
  pip install . && cd ..

# Install dependencies for OpenAI baselines
RUN conda install --channel mpi4py openmpi mpi4py
RUN pip install \
    gym[atari,classic_control] \
    scipy \
    joblib \
    zmq \
    dill \
    azure==1.0.3 \
    progressbar2

RUN apt-get install -y --no-install-recommends swig
RUN git clone https://github.com/pybox2d/pybox2d && \
    cd pybox2d && \
    python setup.py build && \
    python setup.py install

RUN pip install keras==1.2.0