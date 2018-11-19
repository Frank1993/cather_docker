FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has keras 2.1.5 using tensorflow 1.6 backend on python3.6, with numpy, scipy, h5py, opencv-python, Pillow, tqdm, virtualenv and matplotlib" \
      repository="philly/jobs/custom/tensorflow" \
      tag="tf16-keras215-py36" \
      creator="v-dacyin" tooltype="tensorflow" \
      tooltypeversion="1.6" \
      createtime="11/11/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get update
RUN apt-get -y install software-properties-common libcupti-dev

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get -y update && \
    apt-get -y install python3.6
RUN curl https://bootstrap.pypa.io/get-pip.py | sudo python3.6

RUN python3.6 -m pip install --upgrade numpy scipy h5py Keras==2.1.5 opencv-python Pillow tqdm matplotlib virtualenv https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.6.0-cp36-cp36m-linux_x86_64.whl

RUN chmod 777 /usr && \
    mkdir /usr/java/ && \
    chmod 777 /usr/java/ && \
    cd /usr/java/ && \
    wget --no-check-certificate --no-cookies --header "Cookie: oraclelicense=accept-securebackup-cookie" http://download.oracle.com/otn-pub/java/jdk/8u191-b12/2787e4a523244c269598db4e85c51e0c/jdk-8u191-linux-x64.tar.gz && \
    tar -xzvf jdk-8u191-linux-x64.tar.gz

ENV JAVA_HOME=/usr/java/jdk1.8.0_191
ENV JRE_HOME=/usr/java/jdk1.8.0_191/jre
ENV CLASSPATH=.:$JAVA_HOME/lib/ext:$JRE_HOME/lib:$JAVA_HOME/lib:$JRE_HOME/lib:$CLASSPATH
ENV PATH=$JAVA_HOME/bin:$JRE_HOME/bin:$PATH

RUN cd /usr && \
    mkdir /usr/venvs && \
    cd /usr/venvs && \
    virtualenv --no-site-packages -p python2 py2_eval

RUN /bin/bash -c "source /usr/venvs/py2_eval/bin/activate && \
    pip install numpy && \
    deactivate && \
    cd /usr && \
    chmod -R 777 /usr/venvs/"

ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

WORKDIR /workspace
RUN chmod -R a+w /workspace
