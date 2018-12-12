FROM phillyregistry.azurecr.io/philly/jobs/custom/custom-base:custom-base-cuda9.2-openmpi1.10.3

# Labels for the docker
LABEL description="This docker has CUDA 9.2 with support for pytorch-0.4.1 and python-3.6. \
                    It has OpenMPI v1 and needs to be updated to OpenMPI v3" \
      repository="philly/jobs/custom/asr" \
      tag="asr-pytorch0.4.1-py36-cuda92" \
      creator="wigale" \
      tooltype="asr" \
      tooltypeversion="0.4.1" \
      createtime="12/5/2018"

# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
# Horovod requires both tensorflow and pytorch to be installed
ENV PYTORCH_VERSION=0.4.1 \
    NCCL_SOCKET_IFNAME=eth0 \
    NCCL_IB_DISABLE=1 \
    NCCL_IB_CUDA_SUPPORT=0 \
    PATH="/usr/local/mpi/bin:${PATH}" \
    LANG=C.UTF-8

# Install PyTorch
RUN pip install http://download.pytorch.org/whl/cu92/torch-${PYTORCH_VERSION}-cp36-cp36m-linux_x86_64.whl

# Install LM dependencies
RUN pip install cffi \
                tqdm \
                spacy \
                mpi4py \
                ipdb \
                pandas \
                matplotlib \
                py3nvml \
                tensorboard_logger \
                tables \
                graphviz \
                git+https://github.com/szagoruyko/pytorchviz

COPY toolkit-execute /home/job/toolkit-execute
RUN chmod a+x /home/job/toolkit-execute
WORKDIR /home/job/