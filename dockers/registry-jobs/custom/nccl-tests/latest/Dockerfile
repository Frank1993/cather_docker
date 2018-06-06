FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

WORKDIR /

COPY toolkit-execute /home/job/toolkit-execute

RUN apt-get update && apt-get install -y git && \
    git clone https://github.com/NVIDIA/nccl-tests && \
    cd nccl-tests && \
    make MPI=1 MPI_HOME=/usr/local/mpi/
