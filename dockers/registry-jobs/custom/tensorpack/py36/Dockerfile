FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has tensorpack 0.8.8 on tensorflow 1.10.1 using Anaconda5.2(py36)" \
      repository="philly/jobs/custom/tensorpack" \
      tag="tensorpack" \
      creator="v-ling" tooltype="tensorpack" \
      tooltypeversion="0.8.8" \
      createtime="8/31/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN sudo apt-get update && sudo apt-get install -y vim
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

RUN sudo chmod -R 777 /opt/conda
RUN pip install tensorflow-gpu==1.10.1 opencv-python msgpack pycocotools tensorpack \
    ninja torch imgaug tensorboardX torchvision prettytable

