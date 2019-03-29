FROM phillyregistry.azurecr.io/philly/jobs/custom/pytorch:pytorch1.0.1-py36
COPY toolkit-execute /home/job/toolkit-execute
ARG cuda_version=cuda91
ARG PYTHON_VERSION=3.7

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
        libjpeg-dev \
        libpng-dev &&\
    rm -rf /var/lib/apt/lists/*

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN pip install tqdm click tensorboard-logger h5py --upgrade

RUN pip install git+https://riz6bmtj3tjgjafugbx6l34mrhhfbxvs643wihqdz25kklnid56a:x-oauth-basic@dev.azure.com/maluuba/vocab-tools/_git/vocab-tools --upgrade
RUN pip install git+https://riz6bmtj3tjgjafugbx6l34mrhhfbxvs643wihqdz25kklnid56a:x-oauth-basic@dev.azure.com/maluuba/pytorch-utils/_git/pytorch-utils@azureml #helllo4
RUN pip install git+https://riz6bmtj3tjgjafugbx6l34mrhhfbxvs643wihqdz25kklnid56a:x-oauth-basic@dev.azure.com/maluuba/q-gen-pytorch/_git/q-gen-pytorch@joint_qa
