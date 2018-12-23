FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has custom pytorch in python36 with cuda 9.0, and cudnn 7.0. with mpi nccl" \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch1.0.0-py36-mpi-nccl-hvd-apex-nvvl" \
      creator="v-jiarxu" tooltype="pytorch" \
      tooltypeversion="1.0.0" \
      createtime="23/12/2018"

#RUN wget --no-verbose http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl2_2.1.15-1+cuda9.0_amd64.deb && \
#    dpkg -i libnccl2_2.1.15-1+cuda9.0_amd64.deb

RUN apt-get update && apt-get install -y --no-install-recommends \
         git \
		 unzip \
         libnccl2=2.1.15-1+cuda9.0 \
         libnccl-dev=2.1.15-1+cuda9.0 \
         ca-certificates \
         libjpeg-dev \
         libpng-dev &&\
         rm -rf /var/lib/apt/lists/*

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    wget \
    libopenblas-dev \
    libopencv-dev \
    libyaml-dev \
    git && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        vim \
        tmux \
        unzip \
        htop 

# Set timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

# Get Conda-ified Python.
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    sh ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

ENV PATH /opt/conda/bin:$PATH

# Install general libraries
RUN conda install -y python=3.6 numpy pyyaml scipy ipython mkl scikit-learn matplotlib pandas setuptools Cython h5py graphviz
RUN conda clean -ya
RUN conda install -y mkl-include cmake cffi typing
RUN conda install -y -c mingfeima mkldnn

# Install pytorch 
# RUN conda install -y pytorch==1.0.0 torchvision -c pytorch
# Install basic dependencies
RUN conda install -y numpy pyyaml mkl mkl-include setuptools cmake cffi typing
# Add LAPACK support for the GPU
RUN conda install -y -c pytorch magma-cuda90 # or [magma-cuda80 | magma-cuda91] depending on your cuda version
RUN git clone --recursive -b v1.0rc1 https://github.com/pytorch/pytorch && \
    cd pytorch && python setup.py install && cd .. && rm -rf pytorch




# Set CUDA_ROOT
RUN export CUDA_HOME="/usr/local/cuda"

# Install horovod
RUN HOROVOD_GPU_ALLREDUCE=NCCL pip install --no-cache-dir horovod

# Install apex
WORKDIR $STAGE_DIR
RUN pip uninstall -y apex || :
# SHA is something the user can touch to force recreation of this Docker layer, 
# and therefore force cloning of the latest version of Apex
RUN SHA=ToUcHMe git clone https://github.com/NVIDIA/apex.git
WORKDIR $STAGE_DIR/apex
RUN python setup.py install --cuda_ext --cpp_ext
WORKDIR $STAGE_DIR
RUN rm -rf apex

ARG FFMPEG_VERSION=3.4.2
# nvcuvid deps
RUN apt-get update --fix-missing && \
    apt-get install -y libx11-6 libxext6
ENV NVIDIA_DRIVER_CAPABILITIES=video,compute,utility

# minimal ffmpeg from source
RUN apt-get install -y \
      yasm \
      libx264-148 libx264-dev \
      libx265-79 libx265-dev \
      pkg-config && \
    cd /tmp && wget -q http://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.bz2 && \
    tar xf ffmpeg-$FFMPEG_VERSION.tar.bz2 && \
    rm ffmpeg-$FFMPEG_VERSION.tar.bz2 && \
    cd ffmpeg-$FFMPEG_VERSION && \
    ./configure \
    --prefix=/usr/local \
    --disable-static --enable-shared \
    --disable-all --disable-autodetect --disable-iconv \
    --enable-avformat --enable-avcodec --enable-avfilter --enable-avdevice \
    --enable-protocol=file \
    --enable-demuxer=mov,matroska,image2 \
    --enable-bsf=h264_mp4toannexb,hevc_mp4toannexb \
    --enable-gpl --enable-libx264 --enable-libx265 --enable-zlib \
    --enable-indev=lavfi \
    --enable-swresample --enable-ffmpeg \
    --enable-swscale --enable-filter=scale,testsrc \
    --enable-muxer=mp4,matroska,image2 \
    --enable-cuvid --enable-nvenc --enable-cuda \
    --enable-decoder=h264,h264_cuvid,hevc,hevc_cuvid,png,mjpeg,rawvideo \
    --enable-encoder=h264_nvenc,hevc_nvenc,libx264,libx265,png,mjpeg \
    --enable-hwaccel=h264_cuvid,hevc_cuvid \
    --enable-parser=h264,hevc,png && \
    make -j8 && make install && \
    ldconfig && \
    cd /tmp && rm -rf ffmpeg-$FFMPEG_VERSION && \
    apt-get remove -y yasm libx264-dev libx265-dev && \
    apt-get auto-remove -y

# install stub library since driver libs aren't available at image build time
# this is a temporary requirement that will go away in future cuda versions
# libnvcuvid.so was created using the make-stub.sh script
COPY libnvcuvid.so /usr/local/cuda/lib64/stubs

# install nvvl
RUN pip install --upgrade cmake && \
    apt-get install -y pkg-config && \
    cd /tmp && \
    wget https://github.com/NVIDIA/nvvl/archive/master.tar.gz -O nvvl.tar.gz && \
    mkdir nvvl && \
    tar xf nvvl.tar.gz -C nvvl --strip-components 1 && \
    rm nvvl.tar.gz && \
    cd nvvl/pytorch1.0 && \
    python3 setup.py install && \
    pip uninstall -y cmake && \
    apt-get remove -y pkg-config && \
    apt-get autoremove -y

RUN pip install scikit-image psutil torchvision

RUN git clone https://github.com/dukebw/lintel.git && \
    cd lintel && pip install . && \
    cd .. && rm -rf lintel