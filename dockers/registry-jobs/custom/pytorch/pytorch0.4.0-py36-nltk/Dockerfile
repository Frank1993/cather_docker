FROM phillyregistry.azurecr.io/philly/jobs/custom/pytorch:pytorch0.4.0-py36

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 0.4.0 in anaconda python36 with cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch0.4.0-py36-nltk" \
      creator="v-qizhou" tooltype="pytorch" \
      tooltypeversion="0.4.0" \
      createtime="5/29/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN /opt/conda/bin/conda install --name pytorch-py$PYTHON_VERSION nltk && \
    /opt/conda/bin/conda clean -ya
