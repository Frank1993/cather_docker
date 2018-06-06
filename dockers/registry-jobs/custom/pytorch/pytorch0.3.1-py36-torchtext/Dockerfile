FROM phillyregistry.azurecr.io/philly/jobs/custom/pytorch:pytorch0.3.1-py36-nltk

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 0.3.1 with torchtext package(0.2.3) in anaconda python36 with cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch0.3.1-py36-torchtext" \
      creator="v-rux" tooltype="pytorch" \
      tooltypeversion="0.3.1" \
      createtime="6/6/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN pip install torchtext==0.2.3 && \
    pip install spacy && \
    python -m spacy download en
