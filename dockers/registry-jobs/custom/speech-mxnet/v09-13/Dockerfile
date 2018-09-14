FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker mxnet using latest mxnet, copy from zhida" \
      repository="philly/jobs/custom/speech-mxnet" \
      tag="v09-13" \
      creator="v-kahu" tooltype="mxnet" \
      tooltypeversion="1.1.0" \
      createtime="09/13/2018"

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    wget \
    python-dev \
    python-setuptools \
    python-pip \
    python-numpy \
    libgfortran3 \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libopencv-dev \
    libyaml-dev \
    python-tk \
    zip \
    unzip \
    git && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip install --no-cache-dir --upgrade pip==9.0.3 setuptools wheel && \
    pip install --no-cache-dir \
    opencv-python \
    pyyaml \
    easydict \
    Cython \
    Pillow 

RUN pip install numpy matplotlib mock scipy scikit-image shapely Polygon2
RUN  git clone --recursive --branch master https://github.com/YueshangGu/incubator-mxnet  /mxnet

# RUN git clone https://github.com/TuSimple/mx-maskrcnn.git /mx-maskrcnn
# RUN cp /mx-maskrcnn/rcnn/CXX_OP/roi_align-inl.h  /mxnet/src/operator/  
# RUN cp /mx-maskrcnn/rcnn/CXX_OP/roi_align.cc  /mxnet/src/operator/ 
# RUN cp /mx-maskrcnn/rcnn/CXX_OP/roi_align.cu  /mxnet/src/operator/ 

RUN cd /mxnet && make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1 && cd python && sudo python setup.py install

ENV MXNET_CUDNN_AUTOTUNE_DEFALUT 0
ENV MXNET_CUDNN_AUTOTUNE_DEFALUT=0

ENV PATH /usr/local/cuda/bin:${PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}

ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

ENV LIBRARY_PATH /usr/local/cuda/lib64:${LIBRARY_PATH}
ENV LIBRARY_PATH=/usr/local/cuda/lib64:${LIBRARY_PATH}
