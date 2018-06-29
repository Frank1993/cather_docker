FROM phillyregistry.azurecr.io/philly/jobs/custom/pytorch:pytorch0.4.0-py36-nltk

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 0.4.0 with msgpack-python package(0.5.6) in anaconda python36 with cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch0.4.0-py36-msgpack" \
      creator="yuwfan" tooltype="pytorch" \
      tooltypeversion="0.4.0" \
      createtime="23/6/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN pip install msgpack-python==0.5.6 && \
    pip install pandas==0.22.0 && \
    pip install tqdm && \
    pip install spacy && \
    python -m spacy download en
