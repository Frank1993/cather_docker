FROM phillyregistry.azurecr.io/philly/jobs/custom/pytorch:pytorch1.0.0-py36

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 1.0.0 in anaconda python36 with cuda 9.0, cudnn 7.0, and nccl2. + NLTK and spaCy." \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch1.0.0-py36-nlp" \
      creator="v-qizhou" tooltype="pytorch" \
      tooltypeversion="1.0.0" \
      createtime="12/20/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.


RUN /opt/conda/bin/conda install --name pytorch-py$PYTHON_VERSION nltk && \
    /opt/conda/bin/conda clean -ya && \
    pip install -U spacy && \
    python -m spacy download en
