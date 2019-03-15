FROM phillyregistry.azurecr.io/microsoft_pytorch:v1.0.1_gpu_cuda9.0_py36_release_gpuenv_hvd0.15.2

# Labels for the docker
LABEL description="Pytorch Docker used by the acoustic modeling team \
                    inherits from msft-pytorch" \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch-speech" \
      creator="yagaur" \
      tooltype="pytorch-speech" \
      tooltypeversion="1.0.1" \
      createtime="3/14/2019"

# Install AM dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends libsndfile-dev && \
    pip install editdistance tensorboard_logger librosa SoundFile && \
    pip install jupyter psutil && \
    pip install https://github.com/kpu/kenlm/archive/master.zip

COPY toolkit-execute /home/job/toolkit-execute
RUN chmod a+x /home/job/toolkit-execute
WORKDIR /home/job/