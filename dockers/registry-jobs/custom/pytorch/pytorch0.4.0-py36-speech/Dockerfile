FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 0.4.0 in python36 with cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch0.4.0-py36-speech" \
      creator="v-wanke" tooltype="pytorch" \
      tooltypeversion="0.4.0" \
      createtime="05/04/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        vim \
        octave \
        zip \
        unzip && \
    rm -rf /var/lib/apt/lists/*

ARG PYTHON_VERSION=3.6
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy scipy scikit-learn && \
    /opt/conda/bin/conda install -y torchvision cuda90 pytorch=0.4.0 -c pytorch && \
    /opt/conda/bin/conda clean -ya && \
    echo "export PATH=/opt/conda/bin:$PATH" >> ~/.bashrc && \
    echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/include:$LD_LIBRARY_PATH" >> ~/.bashrc && \
    echo "export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/opt/OpenBLAS/lib:$LD_LIBRARY_PATH" >> ~/.bashrc && \
    echo "LC_ALL=C" >> ~/.bashrc

ENV PATH=/opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/include:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/opt/OpenBLAS/lib:$LD_LIBRARY_PATH
ENV LC_ALL=C

RUN pip install --upgrade pip
RUN pip install setuptools six soundfile h5py python-dateutil pysptk tensorboardX librosa
RUN pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.10.0-cp36-cp36m-linux_x86_64.whl

WORKDIR /workspace
RUN chmod -R a+w /workspace
