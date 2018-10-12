FROM wangkenpu/pytorch0.4.0-py36-speech

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 0.4.0 in python36 with cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch0.4.0-py36-speech" \
      creator="v-wanke" tooltype="pytorch" \
      tooltypeversion="0.4.0" \
      createtime="05/04/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

WORKDIR /workspace
RUN chmod -R a+w /workspace
