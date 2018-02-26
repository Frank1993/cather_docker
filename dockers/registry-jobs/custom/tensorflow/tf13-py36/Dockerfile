FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.8.0-cudnn.6

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker contains TF 1.3 using python 3.6 from anaconda, with numpy and h5py" \
      repository="philly/jobs/custom/tensorflow" \
      tag="tf13-py36" \
      creator="retachet" tooltype="tensorflow" \
      tooltypeversion="1.3" \
      createtime="2/26/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

# Install Anaconda for python 3.6
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda3-5.0.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh
ENV PATH /opt/conda/bin:$PATH

# Set CUDA_ROOT
ENV CUDA_ROOT /usr/local/cuda/bin

# Install TensorFlow GPU version.
RUN pip --no-cache-dir install numpy h5py \
          https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.3.0-cp36-cp36m-linux_x86_64.whl