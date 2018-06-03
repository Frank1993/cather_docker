FROM p0jeff/pytorch@sha256:d9992f7d50cb099a8eec55ce29eedfb0b362074c8aa26ceb33cfed2337dfb882
# This docker file inherit from base philly-openmpi and install cuda-aware mpi & pytorch & torchvision & nltk & gym
# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 0.4.0 torchvision in anaconda python36 with cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch-0.4.0-mpi-py36" \
      creator="v-shuzhe" tooltype="pytorch" \
      tooltypeversion="0.4.0" \
      createtime="3/6/2018"

WORKDIR /workspace
RUN chmod -R a+w /workspace

