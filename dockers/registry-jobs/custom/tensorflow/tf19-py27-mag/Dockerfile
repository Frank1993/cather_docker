FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has tensorflow 1.9 using python2.7, with numpy, scipy, h5py, opencv, six, termcolor, tqdm, zmq, tabulate" \
      repository="philly/jobs/custom/tensorflow" \
      tag="tf19-py2.7" \
      creator="juqia" tooltype="tensorflow" \
      tooltypeversion="1.9" \
      createtime="11/2/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.


RUN apt-get update && apt-get -y install python-pip python-dev libopencv-dev python-opencv python-scipy python-numpy python-matplotlib libopencv-dev 

# Install TF
# RUN pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.9.0-cp27-cp27mu-linux_x86_64.whl
RUN pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.11.0-cp27-none-linux_x86_64.whl

RUN python -m pip install --upgrade lmdb six termcolor tqdm msgpack-python msgpack-numpy zmq functools32 subprocess32 tabulate

ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
