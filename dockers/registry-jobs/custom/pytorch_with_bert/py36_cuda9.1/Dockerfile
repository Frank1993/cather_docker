FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.1-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has tensorpack 0.8.8 on tensorflow 1.10.1 using Anaconda5.2(py36), has pytorch_pretrained_bert 0.6.2 on pytorch" \
      repository="philly/jobs/custom/pytorch_with_bert" \
      tag="pytorch_with_bert" \
      creator="t-genl" tooltype="pytorch_pretrained_bert" \
      tooltypeversion="0.6.2" \
      createtime="4/9/2019"
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

RUN sudo chmod -R 777 /opt/conda && \
    pip install tensorflow-gpu==1.10.1 pytorch-pretrained-bert deepdish easydict h5py tensorboardX pysrt opencv-python msgpack pycocotools tensorpack \
        ninja imgaug prettytable   && \
    conda install pytorch torchvision cuda91 -c pytorch

# Install apex
WORKDIR $STAGE_DIR
RUN pip uninstall -y apex || :
RUN git clone https://github.com/NVIDIA/apex.git && cd apex && python setup.py install --cuda_ext --cpp_ext && cd .. \
RUN rm -rf apex
    


