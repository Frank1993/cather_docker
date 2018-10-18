FROM phillyregistry.azurecr.io/philly/jobs/custom/tensorflow:tf110-py36

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has tensorflow 1.10 using python3.6, with spacy and allennlp" \
      repository="philly/jobs/custom/tensorflow" \
      tag="tf110-py36-allennlp" \
      creator="wenwan" tooltype="tensorflow" \
      tooltypeversion="1.10" \
      createtime="10/17/2018"
# Everything above this line is required for the docker.

# Add your personalized features below here.
RUN pip install --upgrade --force-reinstall pip==9.0.3
RUN python -m pip install spacy allennlp
RUN python -m spacy download en


# Set the locale
RUN apt-get clean && apt-get -y update && apt-get install -y locales && locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
RUN locale
