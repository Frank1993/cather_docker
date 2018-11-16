FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 0.3.1 (and other stuff) in anaconda python36 with cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch0.3.1-py36-h5py-mpl-dask-pandas" \
      creator="frcasale" tooltype="pytorch" \
      tooltypeversion="0.4.1" \
      createtime="11/7/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get update
RUN apt-get -y install curl
RUN curl -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh
RUN /opt/conda/bin/conda install -y python=3.6 numpy scipy h5py scikit-learn matplotlib dask pandas
RUN /opt/conda/bin/conda install -y pytorch=0.3.1 -c soumit
RUN /opt/conda/bin/conda install -y torchvision=0.2.1
ENV PATH /opt/conda/bin:$PATH

WORKDIR /workspace
RUN chmod -R a+w /workspace
