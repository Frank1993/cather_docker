FROM phillyregistry.azurecr.io/philly/jobs/custom/generic-docker:py27

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker contains dependencies for the attnGAN project" \
      repository="philly/jobs/custom/msrai" \
      tag="attngan" \
      creator="annelo"  \
      tooltype="torch" \
      tooltypeversion="0.3.1" \
      createtime="5/23/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get update 
RUN apt-get install -y vim

#ADD requirements /root/requirements

ENV PATH /root/.local/bin:$PATH

#RUN pip install -r /root/requirements/requirements.txt  --user


RUN pip install backports.functools-lru-cache==1.5 \
backports.functools-lru-cache==1.5 \
certifi==2018.1.18 \
chardet==3.0.4 \
cycler==0.10.0 \
cython==0.28.1 \
decorator==4.2.1 \
dill==0.2.7.1 \
easydict==1.7 \
idna==2.6 \
matplotlib==2.1.2 \
networkx==2.1 \
nltk==3.2.5 \
numpy==1.14.1 \
pandas==0.22.0 \
pillow==5.0.0 \
pip==9.0.1 \
pyparsing==2.2.0 \
python-dateutil==2.6.1 \
pytz==2018.3 \
pywavelets==0.5.2 \
pyyaml==3.12 \
requests==2.18.4 \
scikit-image==0.13.1 \
scikit-learn==0.19.1 \
scipy==1.0.0 \
screen==1.0.1 \
setuptools==38.5.1 \
six==1.11.0 \
subprocess32==3.2.7 \
torch==0.3.1 \
torchtext==0.2.1 \
torchvision==0.2.1 \
tqdm==4.19.8 \
typing==3.6.4 \
urllib3==1.22 \
wheel==0.30.0 \
azure-storage-blob \
opencv-python \
cffi 

WORKDIR /root
CMD ["/bin/bash"]
