#FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:philly-ubuntu.16.04-cuda.9.0-cudnn.7
#COPY --from=phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.8.0-cudnn.6 /usr/local/cuda-8.0 /usr/local/cuda-8.0

# Labels for the docker
LABEL description="This docker executes Telenav object detection" \
      repository="philly/jobs/custom/telenav" \
      tag="v2" \
      creator="nitrif" \
      createtime="2/8/2019"

ARG PYTHON_VERSION=3.6
ARG CONDA_PYTHON_VERSION=3
ARG CONDA_VERSION=4.5.1
ARG CONDA_DIR=/opt/conda

ENV PATH $CONDA_DIR/bin:$PATH
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_ROOT=$CUDA_HOME
ENV PATH=$PATH:$CUDA_ROOT/bin:$HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_ROOT/lib64

RUN apt-get update && apt-get install -y --allow-downgrades --no-install-recommends --allow-change-held-packages \
        #libcudnn7=7.0.5.15-1+cuda9.0 \
        #libcudnn7-dev=7.0.5.15-1+cuda9.0 \
        libice6 libsm6 libxt6 libxrender1 libfontconfig1 libcups2 graphviz \
        automake autoconf build-essential cmake libopencv* \
        sudo wget curl unzip nano ca-certificates bzip2 git \
        apt-transport-https \
        --yes && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install RabbitMQ
RUN apt-get update
RUN echo "deb http://www.rabbitmq.com/debian/ testing main" >> /etc/apt/sources.list
RUN curl http://www.rabbitmq.com/rabbitmq-signing-key-public.asc | apt-key add -
RUN apt-get install -y rabbitmq-server
RUN rabbitmq-plugins enable --offline rabbitmq_management
RUN chmod -R 777 /var/lib/rabbitmq
RUN chmod -R 777 /var/log/rabbitmq

# Install Elasticsearch
WORKDIR /
RUN apt-get update
#RUN echo "deb https://artifacts.elastic.co/packages/8.x-prerelease/apt stable main" | sudo tee -a /etc/apt/sources.list.d/elastic-8.x.list
#RUN apt-get install -y elasticsearch && apt-get clean
RUN apt-get install -y default-jre
RUN wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-6.6.0.tar.gz
RUN wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-6.6.0.tar.gz.sha512
RUN shasum -a 512 -c elasticsearch-6.6.0.tar.gz.sha512 
RUN tar -xzf elasticsearch-6.6.0.tar.gz
RUN chmod -R 777 ./elasticsearch-6.6.0

RUN apt-get update && \
    wget http://repo.continuum.io/miniconda/Miniconda$CONDA_PYTHON_VERSION-$CONDA_VERSION-Linux-x86_64.sh -O /tmp/anaconda.sh && \
    echo 'export PATH=$CONDA_DIR/bin:$PATH' > /etc/profile.d/conda.sh && \
    /bin/bash /tmp/anaconda.sh -b -p $CONDA_DIR && \
    rm -rf /tmp/* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ARG CONDA=$CONDA_DIR/bin/conda
ARG PIP=$CONDA_DIR/bin/pip

RUN $CONDA update --yes conda && \
    $CONDA install -c pytorch pytorch torchvision &&\
    $CONDA install -c fastai fastai==1.0.36 &&\
    $CONDA install -y h5py Pillow ipywidgets opencv notebook \
           matplotlib pandas=0.22.0 bcolz sympy scikit-image mkl-service scikit-learn libprotobuf protobuf

RUN $PIP install --upgrade pip dask && \
    $PIP install --upgrade statsd openpyxl tqdm elasticsearch python-logstash logstash_formatter \
           git+https://github.com/broadinstitute/keras-resnet \
           amqpstorm==2.4.0 \
           tensorflow_gpu==1.10.1 \
           keras==2.2.4 \
           opencv-python \
           cython \
           pika \
           editdistance python-Levenshtein

RUN $CONDA clean -tisy

ARG USERNAME=job
ARG USERID=1000
# user's home dir should be mapped from EFS
RUN useradd --create-home -s /bin/bash --no-user-group -u $USERID $USERNAME && \
  adduser $USERNAME sudo && \
  echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

WORKDIR /home/job/

RUN mkdir ./output
RUN chmod -R 777 ./output

COPY ./dockers/quality_orientation_image_classifier/python_modules/classification/deploy/keras.json ./dockers/quality_orientation_image_classifier/keras/
COPY ./dockers ./dockers/
RUN chmod -R 777 /home/job/dockers
#RUN chmod 777 /home/job/dockers/*
#RUN chmod 777 /home/job/dockers/*/python_modules
#RUN chmod 777 /home/job/dockers/traffic_signs_detector_yolo/python_modules/object_detection/yolo
COPY ./scripts ./

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod a+x /home/job/toolkit-execute
