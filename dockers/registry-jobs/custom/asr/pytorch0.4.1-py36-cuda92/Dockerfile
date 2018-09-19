FROM phillyregistry.azurecr.io/philly/jobs/custom/custom-base:custom-base-cuda9.2-openmpi1.10.3

# Labels for the docker
LABEL description="This docker has CUDA 9.2 with support for pytorch-0.4.1 \
                    python-3.6 and fairseq. It has OpenMPI v1 and needs to be updated to OpenMPI v3" \
      repository="philly/jobs/custom/asr" \
      tag="pytorch0.4.1-py36-cuda92" \
      creator="wigale" \
      tooltype="asr" \
      tooltypeversion="0.4.1" \
      createtime="9/7/2018"

# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
# Horovod requires both tensorflow and pytorch to be installed
ENV TENSORFLOW_VERSION=1.10.0 \
    PYTORCH_VERSION=0.4.1 \
    NCCL_SOCKET_IFNAME=eth0 \
    NCCL_IB_DISABLE=1 \
    NCCL_IB_CUDA_SUPPORT=0 \
    PATH="/usr/local/mpi/bin:${PATH}" \
    LANG=C.UTF-8

# Install PyTorch
RUN pip install http://download.pytorch.org/whl/cu92/torch-${PYTORCH_VERSION}-cp36-cp36m-linux_x86_64.whl && \
    pip install torchvision

# Install LM dependencies
RUN pip install cffi tqdm spacy mpi4py ipdb pandas matplotlib py3nvml && \
    python -m spacy download en && \
    git clone https://github.com/pytorch/fairseq.git && \
    cd fairseq && \
    python setup.py build && \
    python setup.py develop

# Install AM Team dependencies (separate docker until requirements.txt is supported)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libsndfile-dev && \
    pip install editdistance tensorboard_logger librosa SoundFile

COPY toolkit-execute /home/job/toolkit-execute
RUN chmod a+x /home/job/toolkit-execute
WORKDIR /home/job/