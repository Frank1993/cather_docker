FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:philly-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 1.1.0 in anaconda python36 with cuda 9.0, cudnn 7.0." \
      repository="philly/jobs/custom/pytorch-tts" \
      tag="v1.1-py36-cuda9" \
      creator="t-binliu" tooltype="pytorch" \
      tooltypeversion="1.1.0" \
      createtime="06/20/2019"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get update && \
    apt-get install -y --allow-change-held-packages --no-install-recommends \
            libnccl2=2.4.2-1+cuda9.0 \
            git \
            unzip \
            curl && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

ENV PYTHON_VERSION=3.6

RUN curl -o ~/anaconda.sh -O  https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh  && \
     chmod +x ~/anaconda.sh && \
     ~/anaconda.sh -f -b -p /opt/conda && \
     rm ~/anaconda.sh

ENV PATH /opt/conda/bin/:$PATH

RUN conda install pytorch torchvision cudatoolkit=9.0 -c pytorch && conda clean -ya

RUN pip install --upgrade pip && \
    pip install --upgrade \
                tensorflow==1.13.1 \
                matplotlib==2.1.0 \
                numpy \
                inflect==0.2.5 \
                librosa==0.6.0 \
                scipy==1.0.0 \
                tensorboardX==1.1 \
                Unidecode==1.0.22 \
                pillow

RUN git clone https://github.com/NVIDIA/apex.git \
    && cd apex \
    && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

WORKDIR /workspace
RUN chmod -R a+w /workspace