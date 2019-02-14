FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has tensorflow 1.8 using python36, with numpy, scipy, keras, sklearn, GPy, h5py, Cython, pandas, zmq, opencv-python, IPython and configobj" \
      repository="philly/jobs/custom/tensorflow" \
      tag="tf18-py36-opencv-keras" \
      creator="kulu" tooltype="tensorflow" \
      tooltypeversion="1.8" \
      createtime="02/14/2019"
# Everything above this line is required for the docker.
# Add your personalized features below here.

ENV CUDA_PATH /usr/local/cuda
ENV PATH ${ANACONDA}/bin:${CUDA_PATH}/bin:$PATH
ENV LD_LIBRARY_PATH ${ANACONDA}/lib:${CUDA_PATH}/bin64:$LD_LIBRARY_PATH
ENV C_INCLUDE_PATH ${CUDA_PATH}/include

RUN apt-get update && apt-get install -y --no-install-recommends \
         wget \
         libopencv-dev \
         git \
         libzmq3-dev \
         libhdf5-serial-dev \
         libboost-all-dev 


RUN apt-get update && apt-get install -y --no-install-recommends \
         python3-pip \
         python3-dev \
         python3-tk

RUN pip3 install --upgrade pip 
RUN pip3 install -U setuptools

RUN pip3 install tensorflow-gpu==1.8.0

RUN pip3 install Cython \ 
                scikit-image \
                keras \
                configobj \
                IPython \
                tqdm \
                pandas \
                opencv-python \
                zmq \ 
                h5py \
                cupy-cuda90

# RUN pip3 install --upgrade keras