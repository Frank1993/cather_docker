FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.8.0-cudnn.6

RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF && \
		echo "deb http://download.mono-project.com/repo/ubuntu stable-xenial main" | sudo tee /etc/apt/sources.list.d/mono-official.list
RUN apt-get update
RUN apt-get install -y git build-essential wget vim rpm2cpio cpio mono-complete module-init-tools libgfortran3 libibumad3 libibverbs1 libpmi0 unzip locales zip bc
COPY vimrc /root/.vimrc
RUN locale-gen en_US en_US.UTF-8

ARG MPI_PATH=/usr/local/mvapich
ARG MPI_LIB_PATH=$MPI_PATH/lib64
ARG MVAPICH_URL="http://mvapich.cse.ohio-state.edu/download/mvapich/gdr/2.2/mofed3.2/mvapich2-gdr-2.2-4.cuda8.0.mofed3.2.gnu4.8.5.el7.centos.x86_64.rpm"
ARG MVAPICH_RPM=mvapich2-gdr-cuda8.0-gnu-2.2-3.el7.centos.x86_64.rmp 
RUN wget -q $MVAPICH_URL -O /root/$MVAPICH_RPM && \
	cd /root && rpm2cpio $MVAPICH_RPM |cpio -id && \
	ln -s /root/opt/mvapich2/gdr/no-mcast/2.2/cuda8.0/mpirun/gnu4.8.5  /usr/local/mvapich && \
	cp -rf /usr/local/mvapich/include/* /usr/local/include/ && \
	cp -rf /usr/local/mvapich/lib64/* /usr/local/lib/

ARG NCCL_URL=https://github.com/NVIDIA/nccl/archive/v1.2.3-1+cuda8.0.tar.gz
ARG NCCL_TAR=nccl.1.2.3.tar.gz
ARG NCCL_PATH=/usr/local/nccl
RUN wget -q $NCCL_URL -O /root/$NCCL_TAR &&\
	cd /root && tar -xzf $NCCL_TAR && \
	cd nccl-1.2.3-1-cuda8.0 && mkdir /usr/local/nccl &&\
	make -e PREFIX=/usr/local/nccl install

RUN wget -q https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run -O /root/cuda_8.0.run && \
  cd /root && sh cuda_8.0.run --samples --samplespath=/usr/local/cuda/ --silent &&\
	ln -s /usr/local/cuda/NVIDIA_CUDA-8.0_Samples /usr/local/cuda/samples &&\
	rm /root/cuda_8.0.run

RUN cd /usr/local/cuda/samples/1_Utilities/deviceQuery/ &&\
  make &&\
	cp deviceQuery /usr/local/bin/ &&\
	make clean

RUN echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/cuda.conf && \
	ldconfig
ENV CUDA_DRIVER_PATH=$CUDA_PATH/lib64/stubs/  MPI_PATH=$MPI_PATH MPI_LIB_PATH=$MPI_LIB_PATH LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MPI_LIB_PATH:$NCCL_PATH/lib 

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
