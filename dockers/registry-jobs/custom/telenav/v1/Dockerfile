#FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7
FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:philly-ubuntu.16.04-cuda.9.0-cudnn.7

# Labels for the docker
LABEL description="This docker executes Telenav object detection" \
      repository="philly/jobs/custom/telenav" \
      tag="v1" \
      creator="nitrif" \
      createtime="9/30/2018"

RUN apt-get update
RUN apt-get install -y curl

ARG PYTHON_VERSION=3.6
ARG MINICONDA_PYTHON_VERSION=3
ARG MINICONDA_VERSION=4.4.10
ARG MINICONDA_DIR=/opt/conda
ARG PIP=$MINICONDA_DIR/bin/pip

ENV PATH $MINICONDA_DIR/bin:$PATH
ENV CUDA_ROOT=$CUDA_DIR
ENV PATH=$PATH:$CUDA_ROOT/bin:$HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_ROOT/lib64

RUN apt-get update && apt-get install -y --no-install-recommends --allow-downgrades --allow-change-held-packages \
        libcudnn7=7.0.5.15-1+cuda9.0 \
        libcudnn7-dev=7.0.5.15-1+cuda9.0 \
        libice6 libsm6 libxt6 libxrender1 libfontconfig1 libcups2 graphviz \
        nano \
        unzip \
        bzip2 \
        sudo \
        wget \
        ca-certificates \
        curl \
        --yes && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Conda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda$MINICONDA_PYTHON_VERSION-$MINICONDA_VERSION-Linux-x86_64.sh && \
    bash Miniconda$MINICONDA_PYTHON_VERSION-$MINICONDA_VERSION-Linux-x86_64.sh -p $MINICONDA_DIR -b && \
    rm Miniconda$MINICONDA_PYTHON_VERSION-$MINICONDA_VERSION-Linux-x86_64.sh && \
    conda update -y conda && \
    rm -rf /tmp/* && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

RUN conda install -y python=$PYTHON_VERSION \
                     notebook \
                     Pillow \
                     ipywidgets \
                     scikit-learn \
                     matplotlib \
                     pandas=0.22.0 \
                     h5py && \
    conda clean -tipsy

RUN  conda install -y "testpath<=0.4"

# Install TensorFlow
RUN  $PIP install --upgrade pip && \
     $PIP install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.8.0-cp36-cp36m-linux_x86_64.whl \
            keras==2.1.6 \
            tqdm \
            opencv-python \
            amqpstorm==2.4.0 \
            elasticsearch \
            python-logstash \
            logstash_formatter \
            cython

# Install RabbitMQ
RUN apt-get update
RUN echo "deb http://www.rabbitmq.com/debian/ testing main" >> /etc/apt/sources.list
RUN curl http://www.rabbitmq.com/rabbitmq-signing-key-public.asc | apt-key add -
RUN apt-get install -y rabbitmq-server
RUN rabbitmq-plugins enable --offline rabbitmq_management
RUN chmod -R 777 /var/lib/rabbitmq
RUN chmod -R 777 /var/log/rabbitmq

WORKDIR /home/job/

ADD ms_demo.zip ms_demo.zip
#RUN curl https://cosmos09.osdinfra.net/cosmos/dsdr.local.prod/local/users/nitrif/Telenav/ms_demo.zip -o ms_demo.zip
#RUN wget https://cosmos09.osdinfra.net/cosmos/dsdr.local.prod/local/users/nitrif/Telenav/ms_demo.zip
RUN unzip ms_demo.zip
RUN rm ms_demo.zip
RUN mv ms_telenav_demo apollo

COPY script.py .
COPY mq_consumer_config.json .
COPY start_object_detection_component.sh .
COPY retinanet_config.json apollo/imagerecognition/python_modules/object_detection/retinanet/tools/

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod a+x /home/job/toolkit-execute
RUN chmod -R 777 /home/job

