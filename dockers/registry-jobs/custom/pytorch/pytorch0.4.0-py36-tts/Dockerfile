FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 0.4.0 in anaconda python36 with cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch0.4.0-py36-tts" \
      creator="v-wanke" tooltype="pytorch" \
      tooltypeversion="0.4.0" \
      createtime="05/25/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

ENV PYTHON_VERSION 3.6

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH /opt/conda/bin/:/opt/conda/envs/pytorch-py$PYTHON_VERSION/bin:$PATH

RUN conda create -y --name pytorch-py$PYTHON_VERSION python=$PYTHON_VERSION && \
    conda clean -ya

RUN conda install --name pytorch-py$PYTHON_VERSION numpy scipy && \
    conda clean -ya

RUN conda install --name pytorch-py$PYTHON_VERSION torchvision cuda90 pytorch=0.4.0 -c pytorch && \
    conda clean -ya

RUN apt-get update && apt-get install -y --no-install-recommends \
         libsndfile1 &&\
         rm -rf /var/lib/apt/lists/*

RUN conda install -y --name pytorch-py$PYTHON_VERSION pysoundfile tensorboardx -c conda-forge && \
    conda clean -ya

WORKDIR /workspace
RUN chmod -R a+w /workspace
