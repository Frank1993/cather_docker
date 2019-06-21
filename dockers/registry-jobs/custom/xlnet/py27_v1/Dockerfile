FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:philly-ubuntu.16.04-cuda.10.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has tensorflow-1.13.1, python-2.7" \
      repository="philly/jobs/custom/xlnet" \
      tag="py27_v1" \
      creator="Wutao Lin" \
      tooltype="tensorflow" \
      tooltypeversion="1.13.1" \
      createtime="06/21/2019"

# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN sudo apt-get update
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial

# install python
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda2-2019.03-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
ENV PATH /opt/conda/bin:$PATH
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh"

RUN echo "which pip"
RUN echo "which python"
RUN pip install cython
RUN pip install tensorflow-gpu==1.13.1

# fix io encoding problem
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
    

