FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.8.0-cudnn.6

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker mxnet1.1.0 using anaconda python36 with other toolkits" \
      repository="philly/jobs/custom/mxnet" \
      tag="v110_py27_cuda9" \
      creator="hopeng" tooltype="mxnet" \
      tooltypeversion="0.10.15" \
      createtime="5/30/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    wget \
    libopenblas-dev \
    libopencv-dev \
    libyaml-dev \
    git && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# hopeng: Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        vim \
	tmux
# hopeng: Set timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime


# hopeng: Install atlas
RUN apt-get install -y --no-install-recommends libatlas-base-dev

# hopeng: Install anaconda for python 2.7
RUN mkdir -p /data/home/hopeng/ && cd /data/home/hopeng/ && wget --no-check-certificate https://repo.continuum.io/archive/Anaconda2-5.1.0-Linux-x86_64.sh -O anaconda.sh && \
    /bin/bash anaconda.sh -b -p /data/home/hopeng/anaconda2 && \
    rm anaconda.sh && \
    echo "export PATH=/data/home/hopeng/anaconda2/bin:$PATH" >> ~/.bashrc

ENV PATH /data/home/hopeng/anaconda2/bin:$PATH

# hopeng: initialize workspace
WORKDIR /data/home/hopeng

#RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
#    wget --quiet https://repo.continuum.io/archive/Anaconda2-5.1.0-Linux-x86_64.sh -O ~/anaconda.sh && \
#    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
#    rm ~/anaconda.sh

#ENV PATH /opt/conda/bin:$PATH

RUN pip install --upgrade pip && \
    pip install tqdm \
                EasyDict==1.7 \
                hickle==2.1.0 \
                opencv-python==3.2.0.6 \
                tensorboard==1.0.0a6 \
                shapely==1.6.4 \
                future \
                Cython \ 
                scipy \
                pandas \
                pyyaml \
                json_tricks \
                scikit-image 

# hopeng: compile mxnet
RUN mkdir -p /data/home/hopeng/software/mxnet && cd /data/home/hopeng/software/mxnet && \
    env GIT_SSL_NO_VERIFY=true git clone --recursive https://hwpengms:5b8900cd8d6b1d6a4333db3d3b6c805d1eee8bab@github.com/ancientmooner/mxnet_jy.git -b jiayuan_dev && \
    cd mxnet_jy && git checkout 47c28b3 && make -j $(nproc)

# hopeng: cp mxnet to python
RUN cd /data/home/hopeng/software/mxnet/mxnet_jy && cp lib/* python/mxnet/ && cp -r python/mxnet /data/home/hopeng/anaconda2/lib/python2.7/site-packages/




	

