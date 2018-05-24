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


RUN pip install backports.functools-lru-cache==1.5 --user
RUN pip install backports.functools-lru-cache==1.5 --user
RUN pip install certifi==2018.1.18 --user
RUN pip install chardet==3.0.4 --user
RUN pip install cycler==0.10.0 --user
RUN pip install cython==0.28.1 --user
RUN pip install decorator==4.2.1 --user
RUN pip install dill==0.2.7.1 --user
RUN pip install easydict==1.7 --user
RUN pip install idna==2.6 --user
RUN pip install matplotlib==2.1.2 --user
RUN pip install networkx==2.1 --user
RUN pip install nltk==3.2.5 --user
RUN pip install numpy==1.14.1 --user
RUN pip install pandas==0.22.0 --user
RUN pip install pillow==5.0.0 --user
RUN pip install pip==9.0.1 --user
RUN pip install pyparsing==2.2.0 --user
RUN pip install python-dateutil==2.6.1 --user
RUN pip install pytz==2018.3 --user
RUN pip install pywavelets==0.5.2 --user
RUN pip install pyyaml==3.12 --user
RUN pip install requests==2.18.4 --user
RUN pip install scikit-image==0.13.1 --user
RUN pip install scikit-learn==0.19.1 --user
RUN pip install scipy==1.0.0 --user
RUN pip install screen==1.0.1 --user
RUN pip install setuptools==38.5.1 --user
RUN pip install six==1.11.0 --user
RUN pip install subprocess32==3.2.7 --user
RUN pip install torch==0.3.1 --user
RUN pip install torchtext==0.2.1 --user
RUN pip install torchvision==0.2.0 --user
RUN pip install tqdm==4.19.8 --user
RUN pip install typing==3.6.4 --user
RUN pip install urllib3==1.22 --user
RUN pip install wheel==0.30.0 --user
RUN pip install azure-storage-blob --user

WORKDIR /root
CMD ["/bin/bash"]
