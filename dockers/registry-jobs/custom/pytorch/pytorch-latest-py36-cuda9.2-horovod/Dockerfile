FROM xnming/pytorch-latest-py36-cuda9.2-horovod:torch-1.1.0

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has latest released pytorch with python3.6 and cuda 9.2" \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch-latest-py36-cuda9.2-horovod" \
      creator="v-ximing" \
      tooltype="pytorch" \
      tooltypeversion="latest" \
      createtime="05/15/2019"
# Everything above this line is required for the docker.

# Add your personalized features below here.
RUN git clone https://github.com/pytorch/vision.git ~/torchvision
RUN cd ~/torchvision && python setup.py install
RUN rm -r ~/torchvision

RUN git clone https://github.com/cocodataset/cocoapi.git ~/cocoapi
RUN cd ~/cocoapi/PythonAPI && make install
RUN rm -r ~/cocoapi

WORKDIR /workspace
RUN chmod -R a+w /workspace
