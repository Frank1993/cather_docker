FROM phillyregistry.azurecr.io/philly/jobs/custom/pytorch:pytorch0.4.0-py36-nltk

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has fairseq-0.4.0 with pytorch 0.4 in anaconda python36 with nltk, cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/pytorch" \
      tag="fairseq-0.4.0" \
      creator="tage" tooltype="pytorch" \
      tooltypeversion="0.4.0" \
      createtime="6/21/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN wget https://github.com/pytorch/fairseq/archive/v0.4.0.tar.gz && tar -zxvf v0.4.0.tar.gz
RUN cd /workspace/fairseq-0.4.0 && pip install -r requirements.txt
ENV LANG C.UTF-8
RUN cd /workspace/fairseq-0.4.0 && python setup.py build && python setup.py develop
