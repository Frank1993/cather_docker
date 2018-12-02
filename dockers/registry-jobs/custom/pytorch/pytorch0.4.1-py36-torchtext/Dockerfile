FROM phillyregistry.azurecr.io/philly/jobs/custom/pytorch:pytorch0.4.1-py36

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 0.4.1 and torchtext 0.3.1 in anaconda python36 with cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch0.4.1-py36-torchtext" \
      creator="v-chduan" tooltype="pytorch" \
      tooltypeversion="0.4.1" \
      createtime="11/30/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN /opt/conda/bin/conda install --name pytorch-py$PYTHON_VERSION nltk && \
    /opt/conda/bin/conda install --name pytorch-py$PYTHON_VERSION configargparse=0.13.0 && \
    /opt/conda/bin/conda clean -ya

RUN pip install torchtext==0.3.1
