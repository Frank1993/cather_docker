FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:philly-ubuntu.16.04-cuda.10.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has tensorflow-1.13.1, pytorch-1.0.1 and python-3.5" \
      repository="philly/jobs/custom/pytorch_xlm" \
      tag="cuda10_v2" \
      creator="Wutao Lin" \
      tooltype="pytorch" \
      tooltypeversion="1.0.1" \
      createtime="06/25/2019"

# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN sudo apt-get update && sudo apt-get install -y vim
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial

# fix io encoding problem
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# install python
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
ENV PATH /opt/conda/bin:$PATH
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh"

RUN pip install cython
RUN pip install tensorflow-gpu==1.13.1 deepdish easydict h5py tensorboardX pysrt opencv-python pycocotools tensorpack ninja

# install XLM environments
RUN git clone https://github.com/ericwtlin/XLM.git && cd XLM && pip install cython && cat requirements.txt | xargs pip install && bash install-tools.sh && rm -rf tools/fastBPE/.git/ tools/mosesdecoder/.git/ tools/wikiextractor/.git/ tools/wmt16-scripts/.git/ && mv tools/ /tools/ && cd .. && rm -rf ./XLM

RUN git clone https://github.com/glample/fastBPE.git && cd fastBPE && python setup.py install && cd .. && rm -rf ./fastBPE

# for XLNet
RUN pip install sentencepiece matplotlib
  
 
