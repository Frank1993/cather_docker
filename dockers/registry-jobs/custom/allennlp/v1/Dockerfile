FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod 777 /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 0.4.0 and allennlp 0.5.1" \
      repository="philly/jobs/custom/allennlp" \
      tag="PyLearn-allennlp" \
      creator="chezhu" tooltype="allennlp" \
      tooltypeversion="0.4.0" \
      createtime="7/11/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

run apt-get update && apt-get install -y git && apt-get install -y python-pip python-dev build-essential
run pip install --upgrade pip
run pip install --upgrade virtualenv 

RUN git clone https://github.com/allenai/allennlp.git && \
    pip install -r ./allennlp/requirements.txt && \
    rm ~/.cache/pip/ -rf

env PYTHONPATH="${PYTHONPATH}:/stage/allennlp"
run pip install pandas
run python -m spacy download en
run apt-get install -y vim





