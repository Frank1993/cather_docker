FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has tensorflow 1.9 using python35" \
      repository="philly/jobs/custom/tf19_cuda90_py35" \
      tag="tf19_cuda90_py35" \
      creator="wei" tooltype="tf19_cuda90_py35" \
      tooltypeversion="1.9" \
      createtime="7/17/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get update

RUN apt-get install -y unzip vim tmux git

RUN apt-get update

RUN apt-get install -y --no-install-recommends libcudnn7=7.0.5.15-1+cuda9.0

RUN apt-get -y install software-properties-common libcupti-dev

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get -y update && \
    apt-get -y install python3.5 python3.5-dev python3.5-tk
RUN curl https://bootstrap.pypa.io/get-pip.py | sudo python3.5

RUN python3.5 -m pip install --upgrade \
absl-py==0.2.2 \
astor==0.6.2 \
better-exceptions==0.2.1 \
bleach==1.5.0 \
cloudpickle==0.5.3 \
cycler==0.10.0 \
dask==0.18.1 \
decorator==4.3.0 \
docopt==0.6.2 \
gast==0.2.0 \
grpcio==1.13.0rc3 \
html5lib==0.9999999 \
imageio==2.3.0 \
jsonpickle==0.9.6 \
kiwisolver==1.0.1 \
Markdown==2.6.11 \
matplotlib==2.2.2 \
munch==2.3.2 \
networkx==2.1 \
numpy==1.14.5 \
opencv-python==3.4.1.15 \
pandas==0.23.3 \
Pillow==5.1.0 \
protobuf==3.6.0 \
py-cpuinfo==4.0.0 \
pyparsing==2.2.0 \
python-dateutil==2.7.3 \
pytz==2018.4 \
PyWavelets==0.5.2 \
sacred==0.7.4 \
scikit-image==0.14.0 \
scipy==1.1.0 \
six==1.11.0 \
tensorflow-gpu==1.9.0 \
termcolor==1.1.0 \
toolz==0.9.0 \
tqdm==4.23.4 \
Werkzeug==0.14.1 \
wrapt==1.10.11

#RUN python3.5 -m pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.8.0-cp35-cp35m-linux_x86_64.whl imageio==2.3.0 opencv-python tqdm better-exceptions==0.2.1 sacred==0.7.3 scipy

ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
