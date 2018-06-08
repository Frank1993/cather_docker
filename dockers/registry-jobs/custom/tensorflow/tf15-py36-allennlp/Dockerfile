FROM phillyregistry.azurecr.io/philly/jobs/custom/tensorflow:tf15-py36

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has tensorflow 1.8 using python3.6, with spacy and allennlp" \
      repository="philly/jobs/custom/tensorflow" \
      tag="tf18-py36-allennlp" \
      creator="v-yizwan" tooltype="tensorflow" \
      tooltypeversion="1.8" \
      createtime="5/17/2018"
# Everything above this line is required for the docker.

# Add your personalized features below here.
RUN apt-get -y update && apt-get -y install python3.6-dev
RUN python3.6 -m pip install --upgrade spacy allennlp
RUN python3.6 -m spacy download en


# Set the locale
RUN apt-get clean && apt-get -y update && apt-get install -y locales && locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
RUN locale
