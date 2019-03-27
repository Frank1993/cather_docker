FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 0.4.0 in anaconda python27 with cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch0.4.0-py27-ocr-cuda9" \
      creator="v-fucong" tooltype="pytorch" \
      tooltypeversion="0.4.0" \
      createtime="05/09/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get update && apt-get install -y --no-install-recommends \
         git \
         octave &&\
         rm -rf /var/lib/apt/lists/*

ENV PYTHON_VERSION=2.7

RUN curl -o ~/anaconda.sh -O  https://repo.anaconda.com/archive/Anaconda2-5.1.0-Linux-x86_64.sh  && \
     chmod +x ~/anaconda.sh && \
     ~/anaconda.sh -b -p /opt/conda && \
     rm ~/anaconda.sh 

ENV PATH /opt/conda/bin/:/opt/conda/envs/pytorch-py$PYTHON_VERSION/bin:$PATH
   


RUN conda create -y --name pytorch-py$PYTHON_VERSION python=$PYTHON_VERSION 

RUN conda install --name pytorch-py$PYTHON_VERSION numpy scipy scikit-learn cffi pyyaml ipython opencv cython python-lmdb && conda clean -ya

RUN conda install --name pytorch-py$PYTHON_VERSION torchvision pytorch=0.4.0 cuda90 -c pytorch && conda clean -ya

RUN git clone https://github.com/Fuzesunshine/warp-ctc.git /usr/local/warp-ctc
RUN mkdir /usr/local/warp-ctc/build
RUN cd /usr/local/warp-ctc/build && cmake ..
RUN cd /usr/local/warp-ctc/build && make
RUN cd /usr/local/warp-ctc/build && make install
RUN export CUDA_HOME="/usr/local/cuda"
RUN cd /usr/local/warp-ctc/pytorch_binding && /opt/conda/envs/pytorch-py$PYTHON_VERSION/bin/python setup.py install

RUN chmod 777 /usr/local/warp-ctc/build

WORKDIR /workspace
RUN chmod -R a+w /workspace
