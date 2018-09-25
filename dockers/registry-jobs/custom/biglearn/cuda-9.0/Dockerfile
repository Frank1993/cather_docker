# Tag: nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
# Created: 2018-09-21T01:02:52.285617849Z"
# Label: com.nvidia.cuda.version: 9.0.176
# Label: com.nvidia.cudnn.version: 7.3.0.29
# Ubuntu 16.04.5
FROM nvidia/cuda@sha256:b3bd47c7b675d2f1e1979edc1bd6cc4a6ac989d50ac305f5fcc812962b4a8926

ENV  LANG="en_US.UTF-8" LC_ALL="en_US.UTF-8" LANGUAGE="en_US.UTF-8" LC_TYPE="en_US.UTF-8" TERM=xterm

RUN apt-get update && apt-get install -y apt-transport-https
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF && \
	echo "deb https://download.mono-project.com/repo/ubuntu stable-xenial main" | tee /etc/apt/sources.list.d/mono-official-stable.list

RUN apt-get update
RUN apt-get install -y --no-install-recommends bc  git wget vim rpm2cpio cpio mono-complete unzip zip locales cuda-samples-$CUDA_PKG_VERSION
COPY vimrc /root/.vimrc
RUN locale-gen en_US en_US.UTF-8

ARG CUDA_PATH=/usr/local/cuda-9.0

# install MAVPICH
ARG MPI_PATH=$CUDA_PATH/mvapich
ARG MPI_LIB_PATH=$MPI_PATH/lib64
ARG MVAPICH_URL="http://mvapich.cse.ohio-state.edu/download/mvapich/gdr/2.3rc1/mofed4.2/mvapich2-gdr-mcast.cuda9.0.mofed4.2.gnu4.8.5.slurm-2.3rc1-2.el7.x86_64.rpm"
ARG MVAPICH_RPM=mvapich2-gdr-mcast.cuda9.0.mofed4.2.gnu4.8.5.slurm-2.3rc1-2.el7.x86_64.rpm 
RUN wget -q $MVAPICH_URL -O /root/$MVAPICH_RPM && \
	cd /root && rpm2cpio $MVAPICH_RPM |cpio -id && \
	mv $(realpath -q -m $(find /root/opt -name mpi.h)/../../)  $MPI_PATH
RUN apt-get install -y --no-install-recommends libgfortran3 libnuma1 libibmad5 libibumad3 libpmi0 curl
# END

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MPI_LIB_PATH

RUN cd /usr/local/cuda/samples/1_Utilities/deviceQuery/ &&\
  make &&\
	cp deviceQuery /usr/local/bin/ &&\
	make clean

RUN cd /tmp && curl https://packages.microsoft.com/config/ubuntu/16.04/prod.list > ./microsoft-prod.list && \
    cp ./microsoft-prod.list /etc/apt/sources.list.d/ && \
		curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > microsoft.gpg && \
		cp ./microsoft.gpg /etc/apt/trusted.gpg.d/

RUN apt-get update && apt-get install -y azcopy

# Philly requirements
RUN apt-get install -y rsync jq openssh-server openssh-client sudo net-tools

# Note. there is host name regex checking incompitability of 3.0+ with philly container name. fallback to 2.0 instead
ARG	OPENMPI_VERSIONBASE=1.10 
ENV STAGE_DIR=/root/gpu/install \
		OPENMPI_VERSION=${OPENMPI_VERSIONBASE}.7 \
    OFED_VERSION=4.2-1.2.0.0
ARG	OPENMPI_STRING=openmpi-${OPENMPI_VERSION} 
WORKDIR $STAGE_DIR
RUN wget -q -O - https://download.open-mpi.org/release/open-mpi/v${OPENMPI_VERSIONBASE}/${OPENMPI_STRING}.tar.bz2 | tar -xjf - && \
    cd ${OPENMPI_STRING} && \
    ./configure --prefix=/usr/local/${OPENMPI_STRING} && \
    make -j"$(nproc)" install && \
    rm -rf $STAGE_DIR/${OPENMPI_STRING} && \
    ln -s /usr/local/${OPENMPI_STRING} /usr/local/mpi && \
    test -f /usr/local/mpi/bin/mpic++

# Update environment variables
ENV PATH=/usr/local/mpi/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib64:/usr/local/mpi/lib:$LD_LIBRARY_PATH

# End Philly

# Python
RUN apt-get install -y python3 python3-pip
# make some useful symlinks that are expected to exist
RUN cd /usr/bin \
  && ln -s idle3 idle \
  && ln -s pydoc3 pydoc \
  && ln -s python3 python \
  && ln -s python3-config python-config 

RUN python -m pip install --upgrade pip && \
		python -m pip install pandas sklearn scipy pycparser numpy
#RUN python -m pip install  pythonnet
# End python

ENV ON_PHILLY_VC=true
# Labels for the docker
LABEL description="This docker is built to support .Net environment via mono with sepcifically support for biglearn." \
      repository="philly/jobs/custom/biglearn" \
      tag=".net biglearn" \
      creator="penhe" tooltype="biglearn" \
      tooltypeversion="1.0" \
      createtime="9/21/2018"
# Everything above this line is required for the docker.
# Copy the files to the necessary folder
COPY stdoutParser.py /home/job/stdoutParser.py
COPY toolkit-execute /home/job/toolkit-execute
COPY biglearn-execute /home/job/biglearn-execute

# Add your personalized features below here.
RUN chmod u+x /home/job/toolkit-execute
RUN chmod u+x /home/job/biglearn-execute
RUN chmod u+x /home/job/stdoutParser.py
WORKDIR /home/job
