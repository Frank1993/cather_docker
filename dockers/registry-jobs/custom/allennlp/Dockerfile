FROM allennlp/allennlp:v0.5.1

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


env PYTHONPATH="${PYTHONPATH}:/stage/allennlp"
run pip install pandas
run python -m spacy download en
run apt-get install -y vim





