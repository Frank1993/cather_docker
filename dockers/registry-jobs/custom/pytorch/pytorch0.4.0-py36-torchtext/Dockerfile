FROM phillyregistry.azurecr.io/philly/jobs/custom/pytorch:pytorch0.4.0-py36

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 0.4.0 in anaconda python36 with cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch0.4.0-py36-torchtext" \
      creator="v-rux" tooltype="pytorch" \
      tooltypeversion="0.4.0" \
      createtime="6/11/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

# Set the locale
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# To fulfill requirements from OpenNMT
RUN pip install torchtext==0.2.3 && \
    pip install spacy && \
    pip install tqdm && \
    pip install six && \
    pip install future && \
    python -m spacy download en
