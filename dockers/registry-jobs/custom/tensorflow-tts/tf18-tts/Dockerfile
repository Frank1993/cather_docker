FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has tensorflow 1.8 and all packages Tacotron depends" \
      repository="philly/jobs/custom/tensorflow-tts" \
      tag="tf18-tts" \
      creator="v-haogu" tooltype="tensorflow-tts" \
      tooltypeversion="1.8" \
      createtime="5/26/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.
RUN apt-get update
RUN apt-get -y install python3-pip
RUN apt-get -y install git
RUN python3 -m pip install tensorflow-gpu==1.8.0
RUN python3 -m pip install --upgrade opencv-python h5py six termcolor zmq tabulate
RUN python3 -m pip install --upgrade falcon==1.2.0 inflect==0.2.5 librosa==0.5.1 matplotlib==2.0.2 numpy==1.14.3 scipy==0.19.0 tqdm==4.11.2 Unidecode==0.4.20
