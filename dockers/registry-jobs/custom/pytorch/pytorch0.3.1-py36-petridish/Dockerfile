FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 0.3.1 in anaconda python36 with cuda 9.0, and cudnn 7.0 along with dependencies for petridish. Modified from the one by v-qizhou" \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch0.3.1-py36-petridish2" \
      creator="dedey" tooltype="pytorch" \
      tooltypeversion="0.3.1" \
      createtime="04/02/2019"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get update && apt-get install -y --no-install-recommends \
         git \
         ca-certificates \
         libnccl2 \
         libnccl-dev \
         libjpeg-dev \
         libpng-dev &&\
         rm -rf /var/lib/apt/lists/*

ENV PYTHON_VERSION=3.6

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda create -y --name pytorch-py$PYTHON_VERSION python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl&& \
     /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/envs/pytorch-py$PYTHON_VERSION/bin:$PATH

# Install pytorch 0.3.1
RUN /opt/conda/bin/conda install --name pytorch-py$PYTHON_VERSION cuda90 pytorch=0.3.1 -c pytorch && \
    /opt/conda/bin/conda clean -ya

# Install other dependencies
RUN /opt/conda/bin/conda install --name pytorch-py$PYTHON_VERSION astroid autopep8 backcall certifi cffi colorama decorator graphviz intel-openmp ipython mkl numpy parso pickleshare pip prompt_toolkit pyyaml six matplotlib wheel yaml wrapt matplotlib pillow &&\
    /opt/conda/bin/conda clean -ya

RUN pip install tensorflow
RUN pip install tensorboardX