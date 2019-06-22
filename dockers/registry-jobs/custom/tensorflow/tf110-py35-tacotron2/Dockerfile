FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7
#FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:philly-ubuntu.16.04-cuda.10.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has tensorflow 1.13 with NCCL2, includes: h5py, bz2file, in anaconda python36 with cuda 9.0, and cudnn 7.0." \
      lepository="philly/jobs/custom/tensorflow" \
      tag="tf113-py35-tacotron2" \
      creator="shujliu" tooltype="tensorflow" \
      tooltypeversion="1.13.1" \
      createtime="06/19/2019"

# Add your personalized features below here.


RUN apt-get update && apt-get -y install python3-pip

RUN apt-get install -y libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg libav-tools wget git vim

RUN apt-get install -y python3-numpy
RUN python3 -m pip install falcon==1.2.0 inflect==0.2.5 audioread==2.1.5 librosa==0.6.2 matplotlib==2.0.2  scipy==1.0.0 tqdm==4.11.2 Unidecode==0.4.20 pyaudio==0.2.11 sounddevice==0.3.10 lws keras

RUN python3 -m pip install  tensorflow-gpu==1.10.0




