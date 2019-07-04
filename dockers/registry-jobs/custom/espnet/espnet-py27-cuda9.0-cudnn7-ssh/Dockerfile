FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has espnet with cuda 9.0, and cudnn 7.0." \
      repository="philly/jobs/custom/espnet" \
      tag="espnet-py27-cuda9.0-cudnn7" \
      creator="t-yud" \
      createtime="12/07/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

# Set language
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV CUDA_HOME=/usr/local/cuda

# Set timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime


# Install packages
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    git \
    virtualenv \
    python-dev \
    zlib1g-dev \
    sox \
    libtool \
    subversion \
     build-essential \
    apt-utils \
    cmake \
    g++ \
    gcc \
    git \
    curl \
    wget \
    rsync \
    vim \
    less \
    htop \
    jq \
    tmux \
    unzip \
    sox \
    libtool \
    subversion \
    libatlas3-base \
    gawk  \
    virtualenv\
    emacs\
    flac\
    screen\
    bc
    
WORKDIR /home/
RUN git clone https://github.com/j4ckl1u/espnet.git
RUN cd /home/espnet/tools/ && git clone https://github.com/kaldi-asr/kaldi.git
RUN cd /home/espnet/tools/kaldi/tools/extras && ./install_mkl.sh
RUN cd /home/espnet/tools/ && make all
RUN rm /home/espnet/tools/kaldi-io-for-python/kaldi_io_py.py && ln -s /home/espnet/tools/kaldi-io-for-python/kaldi_io/kaldi_io.py /home/espnet/tools/kaldi-io-for-python/kaldi_io_py.py
RUN cd /home/espnet/tools && . venv/bin/activate && cd kaldi-io-for-python && pip install .
RUN cd /home/espnet/tools && . venv/bin/activate && pip install sentencepiece




ARG uid=200833952
ARG did=200000513
ARG domain=FAREAST
ARG alias=shujliu
ENV DEBIAN_FRONTEND noninteractive
# Install basic packages
 RUN apt-get update \
 && mkdir -p /home/${alias}/.ssh \
 && mkdir -p /etc/sudoers.d/ \
 && chmod 750 /etc/sudoers.d/ \
 && apt-get install openssh-client \
                    vim \
                    tmux \
                    sudo \
                    apt-transport-https \
                    apt-utils \
                    curl \
                    sudo \
                    git \
                    python2.7 \
                    python \
                    git \
                    wget \
                    -y \
# You'll need the below if you're going to run docker with --user as it uses /etc/passwd
# for checking validity.
 && echo "${domain}.domain users:x:${did}:${domain}.${alias}" >> /etc/group \
 && echo "${domain}.${alias}:x: ${uid}:${did}:${alias},,,:/home/${alias}:/bin/bash" >> /etc/passwd \
 && echo "${domain}.${alias}:*:17575:0:99999:7:::" >> /etc/shadow \
 && echo "${domain}.${alias} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${alias} \
# Sudo
 && echo "${domain}.${alias} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/${alias} \
 && echo "" >> /etc/sudoers.d/${alias} \
 && chmod 440 /etc/sudoers.d/${alias} \
# Section to install azure-cli (needs apt-transport-https above)
 && echo "deb [arch=amd64] https://packages.microsoft.com/repos/azure-cli/ xenial main" > /etc/apt/sources.list.d/azure-cli.list \
 && apt-key adv --keyserver packages.microsoft.com --recv-keys 52E16F86FEE04B979B07E28DB02C46DF417A0893 \
 && curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > microsoft.gpg \
 && mv microsoft.gpg /etc/apt/trusted.gpg.d/microsoft.gpg \
 && apt-get update \
 && apt-get install azure-cli -y 
# pip and friends
RUN /usr/bin/wget -q https://bootstrap.pypa.io/get-pip.py \
 && python ./get-pip.py \
 && rm ./get-pip.py 
# Allow ssh via keys
RUN apt-get install openssh-server -y \
 && echo "sshauthcmduser:x:406:406:SSH Authorized Keys Command User:/etc/ssh:/usr/sbin/nologin" >> /etc/passwd 
COPY get_ssh_key.py /etc/ssh/get_ssh_key.py
COPY login.sh /login.sh
RUN chmod 755 /etc/ssh/get_ssh_key.py \
 && chmod 755 /login.sh \
 && sed -i 's/Port\ 22/Port\ 2022/g' /etc/ssh/sshd_config \
 && echo "AuthorizedKeysCommandUser sshauthcmduser" >> /etc/ssh/sshd_config \
 && echo "AuthorizedKeysCommand /etc/ssh/get_ssh_key.py" >> /etc/ssh/sshd_config \
 && update-rc.d ssh enable \
 && pip install pyopenssl azure-keyvault



COPY .bashrc /home/${alias}/.bashrc
RUN chmod u+x /home/${alias}/.bashrc
COPY .profile /home/${alias}/.profile
RUN chmod u+x /home/${alias}/.profile
RUN chown -R ${domain}.${alias} /home/${alias}

EXPOSE 2222
EXPOSE 3000

