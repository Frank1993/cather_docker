FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has tensorflow 15 using python35, with numpy, scipy, h5py, opencv, six, termcolor, tqdm, zmq, tabulate" \
      repository="philly/jobs/custom/tensorflow" \
      tag="tf15-py35" \
      creator="v-wenwa" tooltype="tensorflow" \
      tooltypeversion="1.5" \
      createtime="5/10/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get update && apt-get -y install python3-pip

RUN python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.5.0-cp35-cp35m-linux_x86_64.whl
RUN python3 -m pip install --upgrade numpy scipy
