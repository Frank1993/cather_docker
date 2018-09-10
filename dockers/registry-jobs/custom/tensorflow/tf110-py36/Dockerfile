FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has tensorflow 1.10 using python3.6, with numpy, scipy, h5py, opencv, six, termcolor, tqdm, zmq, tabulate" \
      repository="philly/jobs/custom/tensorflow" \
      tag="tf110-py36" \
      creator="dedey" tooltype="tensorflow" \
      tooltypeversion="1.10" \
      createtime="8/24/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.


# Install python 3.6 via Anaconda

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
        git mercurial subversion

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
        rm ~/anaconda.sh && \
	    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
	        echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
		    echo "conda activate base" >> ~/.bashrc

RUN conda create --name tensorflow_env python=3.6.6

RUN /bin/bash -c "source activate tensorflow_env"

RUN conda install --name tensorflow_env pip numpy six opencv lmdb scipy termcolor==1.1.0 tqdm>4.11.1 msgpack>=0.5.2 msgpack-numpy>=0.4.0 pyzmq>=16 

# Install TF
RUN pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.10.0-cp36-cp36m-linux_x86_64.whl

ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
