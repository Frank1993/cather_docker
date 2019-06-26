FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:philly-ubuntu.16.04-cuda.10.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has cuda 10.0, and cudnn 7.0." \
      repository="philly/jobs/custom/neuraltts" \
      tag="neuraltts-cuda10" \
      creator="wake" \
      createtime="06/26/2019"
# Everything above this line is required for the docker.
# Add your personalized features below here.

# Set language
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install packages and set timezone
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        unzip \
        zip && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
