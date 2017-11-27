FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.14.04-cuda.8.0-cudnn.5

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker executes a bash script only" \
      repository="philly/jobs/custom/base" \
      tag="v1.0" \
      creator="jafrench" tooltype="bash" \
      tooltypeversion="4.3.11(1)-release" \
      created="4/20/2017 13:56"

# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get -y update && \
    apt-get install -y bc
