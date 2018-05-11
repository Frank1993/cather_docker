FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.8.0-cudnn.6

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker contains PyTorch 0.4 using python 2.7 from anaconda, with numpy and h5py, install ctcwarp and bindding with pytorch " \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch0.4.0-py27" \
      creator="wenh" tooltype="pytorch" \
      tooltypeversion="0.4.0" \
      createtime="5/10/2018"
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
RUN conda install pytorch=0.4.0 torchvision -c pytorch

# install lmdb and cmake (in case not installed)
RUN conda install cmake lmdb
# Install WARP-CTC 
RUN apt-get -y update && apt-get install -y git
RUN git clone https://github.com/SeanNaren/warp-ctc.git ~/warp-ctc
RUN mkdir ~/warp-ctc/build
RUN cd ~/warp-ctc/build && cmake ..
RUN cd ~/warp-ctc/build && make

# binding WARPCTC with ptorch

RUN export CUDA_HOME="/usr/local/cuda"
RUN  cd ~/warp-ctc/pytorch_binding && /opt/conda/bin/python2.7 setup.py install

