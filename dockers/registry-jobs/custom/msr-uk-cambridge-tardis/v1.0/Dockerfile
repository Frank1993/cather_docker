FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.8.0-cudnn.6

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="Runs Chainer too" \
      repository="philly/jobs/custom/msr-uk-cambridge-tardis" \
      tag="v1.0" \
      creator="vassilyl" \
      toolkit="chainer" \
      tooltypeversion="3.2"

# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get -y update && \
    apt-get install -y python3-pip
RUN pip3 install numpy==1.13 docopt chainer==3.2 cupy==2.2 h5py scipy PyYAML
