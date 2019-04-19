FROM phillyregistry.azurecr.io/philly/jobs/custom/pytorch:pytorch1.0.0-py36-nlp-bert

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 1.0.0 with anaconda python36 with cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch1.0.0-py36-mag" \
      creator="juqia" tooltype="pytorch" \
      tooltypeversion="1.0.0" \
      createtime="3/27/2019"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN pip install pandas==0.22.0 && \
    pip install tensorboardX tqdm sklearn sklearn_crfsuite matplotlib gensim setuptools && \
    pip install tensorflow && pytorch_pretrained_bert

RUN apt-get update && \
    apt-get install -y \
    vim \
    tmux \
    less \
    unzip

RUN cd /tmp &&  git clone https://github.com/HIT-SCIR/ELMoForManyLangs.git && pip install -e ELMoForManyLangs/
