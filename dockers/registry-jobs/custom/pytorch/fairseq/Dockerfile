FROM phillyregistry.azurecr.io/philly/jobs/custom/pytorch:pytorch1.0.0-py36-nlp

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has fairseq-0.6.0 with pytorch 1.0.0 in anaconda python36 with nltk, cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/pytorch" \
      tag="fairseq-0.6.0" \
      creator="tage" tooltype="pytorch" \
      tooltypeversion="0.6.0" \
      createtime="1/13/2019"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN git clone https://github.com/pytorch/fairseq.git && mv fairseq fairseq-0.6.0
RUN cd /workspace/fairseq-0.6.0 && pip install -r requirements.txt
ENV LANG C.UTF-8
RUN cd /workspace/fairseq-0.6.0 && python setup.py build && python setup.py develop
