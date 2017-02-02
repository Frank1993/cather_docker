FROM ubuntu.14.04-cuda.8.0-cudnn.5.1-openmpi.1.10.3-devel

# Create the home directory for the custom-docker
RUN mkdir /home/custom

# Copy the files to the necessary folder
COPY runCust /home/custom

# Make sure the permissions are correct
RUN chmod u+x /home/custom/runCust

# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get -y update && \
    apt-get install -y bc
