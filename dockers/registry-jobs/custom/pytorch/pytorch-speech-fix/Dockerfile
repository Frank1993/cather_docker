FROM phillyregistry.azurecr.io/microsoft_pytorch_bugfix:master_gpu_cuda9.0_py36_release_2019.03.22_gpuenv_hvd0.16.1

# Labels for the docker
LABEL description="Pytorch Docker used by the acoustic modeling team \
                    inherits from msft-pytorch" \
      repository="philly/jobs/custom/pytorch" \
      tag="pytorch-speech" \
      creator="yagaur" \
      tooltype="pytorch-speech" \
      tooltypeversion="1.0.1" \
      createtime="3/25/2019"

# Install AM dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends libsndfile-dev && \
    pip install editdistance tensorboard_logger librosa SoundFile && \
    pip install jupyter psutil && \
    pip install https://github.com/kpu/kenlm/archive/master.zip

COPY toolkit-execute /home/job/toolkit-execute
RUN chmod a+x /home/job/toolkit-execute
WORKDIR /home/job/
