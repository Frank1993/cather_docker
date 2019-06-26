FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has cuda9." \
      repository="philly/jobs/custom/neuraltts" \
      tag="neuraltts-cuda9" \
      creator="yanqliu" \
      createtime="05/28/2019"
# Everything above this line is required for the docker.
# Add your personalized features below here.

# Set language
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Set timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

# Install packages
RUN apt-get update && apt-get install -y \
    wget \
    unzip

RUN apt-get install zip

# wget espnet and kaldi
