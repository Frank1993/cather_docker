#FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:philly-ubuntu.16.04-cuda.9.0-cudnn.7
#FROM phillyregistry.azurecr.io/philly/jobs/toolkit/tensorflow:community-gpu-1.12.0-cp35
FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has CUDA and Mono to support running SmartFlow based job" \
      repository="philly/jobs/custom/smartflow" \
      tag="v0.1" \
      creator="yulanyan" \
	  tooltype="smartflow" \
      tooltypeversion="v0.1" \
      createtime="3/24/2019"

# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF
RUN apt-get update
RUN apt install apt-transport-https ca-certificates
RUN echo "deb https://download.mono-project.com/repo/ubuntu stable-xenial main" | tee /etc/apt/sources.list.d/mono-official-stable.list
RUN apt update
RUN apt install -y mono-devel 

#ADD ./Code/Transformer/bin /DeepGen/Code/Transformer/bin
#ADD ./Data /DeepGen/Data

#ENV MONO_LOG_LEVEL=debug
