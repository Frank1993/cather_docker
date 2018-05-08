FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.8.0-cudnn.5

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 0.3.1 in anaconda python27 with cuda 8.0, and cudnn 5.0." \
      repository="philly/jobs/custom/pytorch" \
      tag="mahjong-rl" \
      creator="qiwye" tooltype="pytorch" \
      tooltypeversion="0.3.1" \
      createtime="4/10/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get update && apt-get install -y --no-install-recommends \
         git \
         ca-certificates \
         libnccl2 \
         libnccl-dev \
         libjpeg-dev \
         mono-complete \
         tzdata \
         libpng-dev &&\
         rm -rf /var/lib/apt/lists/*

ENV PYTHON_VERSION=3.6
ENV PYTHON2_VERSION=2.7

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda create -y --name pytorch-py$PYTHON_VERSION python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl&& \
     /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/envs/pytorch-py$PYTHON_VERSION/bin:$PATH

RUN /opt/conda/bin/conda update -n base conda

RUN /opt/conda/bin/conda install --name pytorch-py$PYTHON_VERSION cuda90 pytorch=0.3.1 torchvision -c pytorch

RUN /opt/conda/bin/conda install --name pytorch-py$PYTHON_VERSION pydot graphviz 

RUN /opt/conda/bin/conda install --name pytorch-py$PYTHON_VERSION graphviz -c anaconda

RUN /opt/conda/bin/conda install --name pytorch-py$PYTHON_VERSION pyro4 -c conda-forge

RUN /opt/conda/bin/conda create -y --name theano-py$PYTHON2_VERSION python=$PYTHON2_VERSION numpy pyyaml scipy ipython mkl theano=0.9 &&\
    /opt/conda/bin/conda clean -ya

RUN /opt/conda/bin/conda install --name theano-py$PYTHON2_VERSION mkl-service -c anaconda

COPY sparse_data_parser-3.5/* /home/data_parser/

RUN cd /home/data_parser && \
    python setup.py build && \
    python setup.py install && \
    python -c "from sparse_data_parser import test_parse_sparse_line; test_parse_sparse_line()"
RUN pip install tensorboardX

COPY sparse_data_parser-2.7/* /home/data_parser_2/

RUN ln -s /opt/conda/envs/theano-py$PYTHON2_VERSION/bin/python /usr/local/bin/py2
RUN ln -s /opt/conda/envs/theano-py$PYTHON2_VERSION/bin/pip /usr/local/bin/pip-2

RUN cd /home/data_parser_2 && \
    py2 setup.py build && \
    py2 setup.py install && \
    py2 -c "from sparse_data_parser import test_parse_sparse_line; test_parse_sparse_line()"

RUN pip-2 install pydot graphviz pyyaml pyro4 'theano==0.9' https://github.com/Lasagne/Lasagne/archive/master.zip

WORKDIR /workspace
RUN chmod -R a+w /workspace
