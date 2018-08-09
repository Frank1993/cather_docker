FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# This docker file inherit from base philly-openmpi and install cuda-aware mpi & pytorch & torchvision & nltk & gym
# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 0.5.0 torchvision in anaconda python36 with cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch-0.5.0-gloo-py36" \
      creator="v-shuzhe" tooltype="pytorch" \
      tooltypeversion="0.5.0a" \
      createtime="4/6/2018"

RUN apt-get update && apt-get install -y --no-install-recommends \
         git \
         vim \
         ca-certificates \
         libnccl2 \
         libnccl-dev \
         libjpeg-dev \
         libpng-dev \
         unzip &&\
         rm -rf /var/lib/apt/lists/*

ENV PYTHON_VERSION=3.6

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH 
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN cd ~ &&\
    conda install -y numpy pyyaml scipy ipython mkl mkl-include setuptools cmake cffi typing && \
    conda install -y cuda90 -c pytorch&& \
    conda install -y -c mingfeima mkldnn &&\
    conda install -y -c pytorch magma-cuda90 &&\
    conda install -y cffi tqdm &&\
    conda install -y opencv matplotlib && \
# conda search openmpi -c conda-forge -----> only have 3.1.0 version openmpi, seems this version can not pass philly's hostname validation .
# conda install -y -c conda-forge openmpi &&\
# conda install -y -c pytorch pytorch=0.4.0 torchvision nltk && \
    git clone --recursive https://github.com/pytorch/pytorch && cd pytorch && python setup.py install &&\
    cd ~ && git clone --recursive https://github.com/pytorch/vision.git && cd vision && python setup.py install &&\
#    conda install -y -c pytorch torchvision nltk &&\
    conda install -y ninja &&\
    git clone https://github.com/zhanghang1989/PyTorch-Encoding && cd PyTorch-Encoding && python setup.py install &&\
    python scripts/prepare_ade20k.py && python scripts/prepare_pascal.py && python scripts/prepare_pcontext.py &&\
    conda clean -ya 

#RUN cd ~/.encoding/data &&\
#    wget http://images.cocodataset.org/zips/train2017.zip && unzip train2017.zip &&\
#    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip annotations_trainval2017.zip &&\
#    wget http://images.cocodataset.org/zips/val2017.zip && unzip val2017.zip &&\
#    wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip && unzip stuff_annotations_trainval2017.zip &&\
#    wget http://images.cocodataset.org/zips/test2017.zip && unzip test2017.zip &&\
#    git clone https://github.com/cocodataset/cocoapi && cd cocoapi/PythonAPI/ && conda install Cython && python setup.py install &&\
#    cd ../../ && rm -rf cocoapi/



