FROM phillyregistry.azurecr.io/philly/jobs/custom/pytorch:v0.4.0-py36-vcpose

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 0.4.1 with cudnn off for BatchNorm in anaconda python36 with cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/pytorch" \
      tag="v0.4.1-py36-vcpose-bn-cudnn-off" \
      creator="bixi" tooltype="pytorch" \
      tooltypeversion="0.4.1" \
      createtime="28/09/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN sed -i "1254s/torch\.backends\.cudnn\.enabled/False/g" /opt/conda/lib/python3.6/site-packages/torch/nn/functional.py

WORKDIR /workspace
RUN chmod -R a+w /workspace
