FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 0.4.0 in anaconda python36 with cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch0.4.0-py36-speech" \
      creator="v-wanke" tooltype="pytorch" \
      tooltypeversion="0.4.0" \
      createtime="05/04/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get update && apt-get install -y --no-install-recommends \
         git \
         octave &&\
         rm -rf /var/lib/apt/lists/*

ENV PYTHON_VERSION 3.6

RUN curl -o ~/anaconda.sh -O https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh && \
    chmod +x ~/anaconda.sh && ~/anaconda.sh -b -p /opt/conda && rm ~/anaconda.sh

ENV PATH /opt/conda/bin/:/opt/conda/envs/pytorch-py$PYTHON_VERSION/bin:$PATH

RUN conda create -y --name pytorch-py$PYTHON_VERSION python=$PYTHON_VERSION && \
    conda clean -ya
RUN conda install numpy scipy scikit-learn

RUN conda install --name pytorch-py$PYTHON_VERSION torchvision cuda90 pytorch=0.4.0 -c pytorch && \
    conda clean -ya

WORKDIR /workspace
RUN chmod -R a+w /workspace
