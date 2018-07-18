FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.8.0-cudnn.6

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker contains PyTorch 0.3 using python 2.7 from anaconda, with numpy and h5py" \
      repository="philly/jobs/custom/pytorch" \
      tag="pt03-py27" \
      creator="alsordon" tooltype="pytorch" \
      tooltypeversion="0.3" \
      createtime="7/17/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

# Install Anaconda for python 2.7
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

# Install general libraries
RUN conda install -y python=2.7 numpy pyyaml scipy ipython mkl scikit-learn matplotlib pandas setuptools Cython h5py
RUN conda clean -ya

# Set CUDA_ROOT
ENV CUDA_ROOT /usr/local/cuda/bin

# Install pytorch GPU version.
RUN conda install pytorch torchvision -c pytorch
