FROM phillyregistry.azurecr.io/philly/jobs/custom/pytorch:pytorch0.3.1-py36-nltk

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 0.3.1 with allennlp package in anaconda python36 with cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch0.3.1-py36-allennlp" \
      creator="v-qizhou" tooltype="pytorch" \
      tooltypeversion="0.3.1" \
      createtime="4/17/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN /opt/conda/bin/conda install --name pytorch-py$PYTHON_VERSION nltk && \
    /opt/conda/bin/conda clean -ya
    
RUN git clone https://github.com/allenai/allennlp.git && \
    pip install -r /workspace/allennlp/requirements.txt && \
    python -m nltk.downloader punkt && \
    spacy download en_core_web_sm && \
    rm ~/.cache/pip/ -rf
