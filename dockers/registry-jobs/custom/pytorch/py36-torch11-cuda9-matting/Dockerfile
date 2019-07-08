FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 1.1.0 in anaconda python36 with cuda 9.0, cudnn 7.0" \
      repository="philly/jobs/custom/pytorch" \
      tag="py36-torch11-cuda9-matting" \
      creator="v-zhli8" tooltype="pytorch" \
      tooltypeversion="1.1.0" \
      createtime="07/08/2019"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN sudo apt-get update && sudo apt-get install -y vim tmux zip eog
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

RUN echo "alias ll=0'ls -alF'" >> ~/.bashrc && \
    echo "alias la='ls -A'" >> ~/.bashrc && \
    echo "alias l='ls -CF'" >> ~/.bashrc

RUN sudo chmod -R 777 /opt/conda

RUN pip install opencv-python tqdm jupyter

RUN conda install pytorch=1.1.0 torchvision cudatoolkit=9.0 -c pytorch

WORKDIR /workspace
RUN chmod -R a+w /workspace

