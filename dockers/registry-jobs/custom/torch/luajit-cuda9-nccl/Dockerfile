FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has torch 7 (luajit) with tds and bit32 with cuda9, cudnn7 and nccl, where OpenNMT can be run." \
      repository="philly/jobs/custom/torch" \
      tag="luajit-cuda9-nccl" \
      creator="tage" tooltype="torch" \
      tooltypeversion="torch7" \
      createtime="5/31/2018"

# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get -y update && apt-get install -y git
RUN git clone https://github.com/torch/distro.git ~/torch --recursive
RUN cd ~/torch; bash install-deps;
ENV TORCH_NVCC_FLAGS "-D__CUDA_NO_HALF_OPERATORS__"
RUN cd ~/torch; ./install.sh
RUN cd ~/; chmod 777 .
ENV PATH $PATH:/root/torch/install/bin
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/root/torch/install/lib
RUN luarocks install tds
RUN luarocks install bit32
RUN cd ~/; git clone https://github.com/NVIDIA/nccl.git
RUN cd ~/nccl; make CUDA_HOME=/usr/local/cuda test
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/root/nccl/build/lib
RUN luarocks install nccl
