FROM phillyregistry.azurecr.io/philly/jobs/custom/pytorch:pytorch0.4.0-py36-nltk

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 0.4.0 with anaconda python36 with cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch0.4.0-py36-mag" \
      creator="juqia" tooltype="pytorch" \
      tooltypeversion="0.4.0" \
      createtime="07/18/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN pip install pandas==0.22.0 && \
    pip install tensorboardX

RUN apt-get update && \
    apt-get install -y \
    vim \
    tmux
