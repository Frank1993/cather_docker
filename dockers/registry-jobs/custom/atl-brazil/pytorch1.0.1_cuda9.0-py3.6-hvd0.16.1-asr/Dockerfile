FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

ENV NCCL_VERSION=2.3.7-1+cuda9.0

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 1.0.1, cuda 9.0, python 3.6, horovod 0.16.1, apex, and asr-specific packages" \
    repository="philly/jobs/custom/atl-brazil" \
    tag="pytorch1.0.1_cuda9.0-py3.6-hvd0.16.1-asr" \
    creator="v-igquin" tooltype="pytorch" \
    tooltypeversion="1.0.1" \
    createtime="4/17/2019"
# Everything above this line is required for the docker.
# Add your personalized features below here.

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    bmon \
    bzip2 \
    ca-certificates \
    cmake \
    curl \
    git \
    htop \
    iotop \
    libjpeg-dev \
    libnccl-dev=${NCCL_VERSION} \
    libnccl2=${NCCL_VERSION} \
    libpng-dev \
    libsox-dev \
    libsox-fmt-all \
    sox \
    tree \
    vim \
    wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*


# Configure NCCL and OpenMPI to run good defaults:
RUN echo "hwloc_base_binding_policy = none" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "rmaps_base_mapping_policy = slot" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "btl = ^openib" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "btl_tcp_if_exclude = docker0,lo" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo NCCL_DEBUG=INFO >> /etc/nccl.conf && \
    echo NCCL_SOCKET_IFNAME=^docker0 >> /etc/nccl.conf && \
    echo NCCL_IB_HCA=mlx5_0,mlx5_2 >> /etc/nccl.conf

RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH

# Change to bash
SHELL ["/bin/bash", "-c"]

## Updating conda and pip
RUN conda update --yes -n base conda

RUN pip install pip --upgrade && \
    pip install wget

# Install pytorch
RUN conda install -c pytorch pytorch=1.0.1 cudatoolkit=9.0

RUN conda install ipython

# Install Horovod, temporarily using CUDA stubs
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod==0.16.1 && \
    ldconfig

# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

RUN git clone https://github.com/NVIDIA/apex.git && \
    cd apex && \
    python setup.py install --cuda_ext --cpp_ext && \
    cd .. && \
    rm -rf apex

# install dependencies
RUN pip install \
    editdistance \
    git+https://github.com/parlance/ctcdecode.git \
    git+https://github.com/pytorch/audio.git \
    git+https://github.com/pytorch/ignite.git \
    git+https://github.com/pytorch/vision.git \
    num2words \
    tqdm \
    unidecode

RUN conda install -y seaborn \
    matplotlib \
    h5py \
    scikit-learn \
    jupyter

RUN conda uninstall -y --force pillow pil jpeg libtiff && \
    pip   uninstall -y         pillow pil jpeg libtiff && \
    conda install -y -c conda-forge libjpeg-turbo && \
    CFLAGS="${CFLAGS} -mavx2" pip install --upgrade \
            --no-cache-dir --force-reinstall --no-binary \
            :all: --compile pillow-simd && \
    conda install -y jpeg libtiff

# install warp-CTC
ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
RUN git clone https://github.com/SeanNaren/warp-ctc.git && \
    cd warp-ctc && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    cd .. && \
    cd pytorch_binding && \
    CUDA_HOME="/usr/local/cuda" python setup.py install && \
    cd .. && \
    cp build/libwarpctc.so /usr/local/lib && \
    cd .. && \
    rm -rf warp-ctc

# install test requirements
RUN pip install --upgrade pytest \
    codecov \
    pytest-cov \
    pytest-flake8

RUN conda clean -a -y

# Setting permissions
RUN chmod -R ugo+rw /opt/conda