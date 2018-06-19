FROM phillyregistry.azurecr.io/philly/jobs/custom/tensorflow:tf14-py35

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has tensorflow 14 using python35, with nltk and spacy." \
      repository="philly/jobs/custom/tensorflow" \
      tag="tf14-py35-spacy" \
      creator="v-qizhou" tooltype="tensorflow" \
      tooltypeversion="1.4.0" \
      createtime="6/19/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN python3 -m pip install nltk tqdm scikit-learn h5py ujson keras ftfy joblib spacy

RUN python3 -m spacy download en
