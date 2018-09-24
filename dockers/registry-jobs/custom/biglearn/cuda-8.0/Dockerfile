FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.8.0-cudnn.6

ENV  LANG="en_US.UTF-8" LC_ALL="en_US.UTF-8" LANGUAGE="en_US.UTF-8" LC_TYPE="en_US.UTF-8" TERM=xterm

RUN apt-get update && apt-get install -y apt-transport-https
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF && \
	echo "deb https://download.mono-project.com/repo/ubuntu stable-xenial main" | tee /etc/apt/sources.list.d/mono-official-stable.list

RUN apt-get update
RUN apt-get install -y --no-install-recommends bc  git wget vim rpm2cpio cpio mono-complete unzip zip locales cuda-samples-$CUDA_PKG_VERSION

COPY vimrc /root/.vimrc
RUN locale-gen en_US en_US.UTF-8

ARG CUDA_PATH=/usr/local/cuda-8.0
ARG MPI_PATH=$CUDA_PATH/mvapich
ARG MPI_LIB_PATH=$MPI_PATH/lib64
ARG MVAPICH_URL="http://mvapich.cse.ohio-state.edu/download/mvapich/gdr/2.2/mofed3.2/mvapich2-gdr-2.2-4.cuda8.0.mofed3.2.gnu4.8.5.slurm.el7.centos.x86_64.rpm"
ARG MVAPICH_RPM=mvapich2-gdr-2.2-4.cuda8.0.mofed3.2.gnu4.8.5.slurm.el7.centos.x86_64.rpm 
RUN wget -q $MVAPICH_URL -O /root/$MVAPICH_RPM && \
	cd /root && rpm2cpio $MVAPICH_RPM |cpio -id && \
	ln -s $(realpath -q -m $(find /root/opt -name mpi.h)/../../) $MPI_PATH 

ARG NCCL_URL=http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
RUN wget -q $NCCL_URL -O /root/nccl_src.deb &&\
	cd /root && dpkg -i nccl_src.deb && \
	apt-get update && apt-get install libnccl2=2.2.13-1+cuda8.0 libnccl-dev=2.2.13-1+cuda8.0

RUN cd /usr/local/cuda/samples/1_Utilities/deviceQuery/ &&\
  make &&\
	cp deviceQuery /usr/local/bin/ &&\
	make clean

RUN cd /tmp && curl https://packages.microsoft.com/config/ubuntu/16.04/prod.list > ./microsoft-prod.list && \
    cp ./microsoft-prod.list /etc/apt/sources.list.d/ && \
		curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > microsoft.gpg && \
		cp ./microsoft.gpg /etc/apt/trusted.gpg.d/

RUN apt-get update && apt-get install -y --allow-unauthenticated azcopy

RUN echo "$CUDA_PATH/lib64" >> /etc/ld.so.conf.d/cuda-8.0.conf && \
		echo "$MPI_LIB_PATH" >> /etc/ld.so.conf.d/cuda-8.0.conf \
		&&	ldconfig
ENV CUDA_DRIVER_PATH=$CUDA_PATH/lib64/stubs/  

ENV ON_PHILLY_VC=true
# Labels for the docker
LABEL description="This docker is built to support .Net environment via mono with sepcifically support for biglearn." \
      repository="philly/jobs/custom/biglearn" \
      tag=".net biglearn" \
      creator="penhe" tooltype="biglearn" \
      tooltypeversion="1.0" \
      createtime="6/25/2018"
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
