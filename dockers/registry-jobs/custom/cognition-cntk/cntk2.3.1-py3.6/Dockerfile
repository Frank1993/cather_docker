FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.8.0-cudnn.6

RUN mkdir /home/job
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod a+rwx /home/job/toolkit-execute

LABEL description="This docker is for the program synthesis projects of MSR AI Cognition. It has Python 3.6, CNTK 2.3.1, and a bunch of standard numerical pip packages." \
      repository="philly/jobs/custom/cognition-cntk" \
      tag="cntk2.3.1-py3.6" \
      creator="polozov" tooltype="cntk"

RUN apt-get -y update && \
    apt-get -y --fix-missing install software-properties-common
RUN sudo add-apt-repository ppa:deadsnakes/ppa
RUN apt-get -y update && \
    apt-get -y install python3.6
RUN curl https://bootstrap.pypa.io/get-pip.py | sudo python3.6

RUN python3.6 -m pip install https://cntk.ai/PythonWheel/GPU/cntk-2.3.1-cp36-cp36m-linux_x86_64.whl
RUN python3.6 -m pip install numpy scipy jsonlines dill tqdm

