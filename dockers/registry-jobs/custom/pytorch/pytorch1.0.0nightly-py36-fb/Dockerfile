FROM phillyregistry.azurecr.io/philly/jobs/custom/pytorch:pytorch1.0.0a0-py36

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker pytorch1.0 using anaconda py36 with other toolkits" \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch1.0.0nightly-py36-fb" \
      creator="v-bojian" tooltype="pytorch" \
      tooltypeversion="1.0.0nightly" \
      createtime="12/10/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Set timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

RUN conda install -y ipython
RUN pip install ninja yacs cython matplotlib opencv-python

# Install TorchVision master
RUN git clone https://github.com/pytorch/vision.git \
 && cd vision \
 && python setup.py install

# install pycocotools
RUN git clone https://github.com/cocodataset/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && python setup.py build_ext install