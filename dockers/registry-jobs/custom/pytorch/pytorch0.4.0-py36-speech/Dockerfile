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
        software-properties-common \
        python3-dev \
        python3-pip \
        zip \
        unzip \
        vim \
        octave && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHON_VERSION=3.6

RUN add-apt-repository ppa:jonathonf/python-3.6 && \
    apt-get update && apt-get install -y python3.6 && \
    cd /usr/bin && rm python3 && ln -s python3.6m python3
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.6-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install setuptools six
RUN pip3 install numpy scipy scikit-learn cython
RUN pip3 install pysptk
RUN pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.0-cp36-cp36m-linux_x86_64.whl
RUN pip3 install torchvision
RUN pip3 install tensorboardX
RUN pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.8.0-cp36-cp36m-linux_x86_64.whl

WORKDIR /workspace
RUN chmod -R a+w /workspace
