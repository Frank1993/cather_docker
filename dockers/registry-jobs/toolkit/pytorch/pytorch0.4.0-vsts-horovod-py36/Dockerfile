FROM phillyregistry.azurecr.io/philly/jobs/toolkit/pytorch:pytorch0.4.0-vsts-py36
ARG PYTHON_VERSION=36
ARG PYTORCH_VERSION=0.4.0
# Labels for the docker
LABEL description="This docker has horovod-0.13.11 with support for pytorch-$PYTORCH_VERSION, python-$PYTHON_VERSION and openmpi 1.10.3." \
      repository="philly/jobs/toolkit/pytorch" \
      tag="pytorch$PYTORCH_VERSION-vsts-horovod-py$PYTHON_VERSION" \
      creator="tix" \
      tooltype="pytorch+horovod" \
      tooltypeversion="$PYTORCH_VERSION" \
      createtime="8/10/2018"

# Install Horovod
RUN apt-get install libxml2-dev libxslt-dev -y && \
    pip install py3nvml && \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITHOUT_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod==0.13.11 && \
    ldconfig

# Configure NCCL with INFO level:
#   -x NCCL_DEBUG=INFO
RUN echo NCCL_DEBUG=INFO >> /etc/nccl.conf

COPY toolkit-execute /home/job/toolkit-execute
RUN chmod a+x /home/job/toolkit-execute
WORKDIR /home/job/
COPY entrypoint.sh /home/job/entrypoint.sh
RUN chmod 755 /home/job/entrypoint.sh
ENTRYPOINT ["/home/job/entrypoint.sh"]