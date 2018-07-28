FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod 777 /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 0.4.0 and allennlp 0.5.1" \
      repository="philly/jobs/custom/allennlp" \
      tag="PyLearn-allennlp" \
      creator="chezhu" tooltype="allennlp" \
      tooltypeversion="0.4.0" \
      createtime="7/11/2018"
# Everything above this line is required for the docker.
# Add your personalized features below here.

RUN apt-get update && apt-get install -y --no-install-recommends apt-transport-https 
RUN apt-get install -y git build-essential wget vim rpm2cpio cpio  module-init-tools libgfortran3 libibumad3 libibverbs1 libpmi0 unzip locales libssl-dev


# ensure local python is preferred over distribution python
ENV PATH /usr/local/bin:$PATH

# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
# extra dependencies (over what buildpack-deps already includes)
RUN apt-get update && apt-get install -y --no-install-recommends libgtk2.0-dev tk-dev

ENV GPG_KEY 0D96DF4D4110E5C43FBFB17F2D347EA6AA65421D
ENV PYTHON_VERSION 3.6.6

# Disable crypt.h for pythonnet installation
RUN set -ex \
  \
  && wget -O python.tar.xz "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz" \
  && wget -O python.tar.xz.asc "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz.asc" \
  && export GNUPGHOME="$(mktemp -d)" \
  && gpg --keyserver ha.pool.sks-keyservers.net --recv-keys "$GPG_KEY" \
  && gpg --batch --verify python.tar.xz.asc python.tar.xz \
  && { command -v gpgconf > /dev/null && gpgconf --kill all || :; } \
  && rm -rf "$GNUPGHOME" python.tar.xz.asc \
  && mkdir -p /usr/src/python \
  && tar -xJC /usr/src/python --strip-components=1 -f python.tar.xz \
  && rm python.tar.xz \
  \
  && cd /usr/src/python \
  && gnuArch="$(dpkg-architecture --query DEB_BUILD_GNU_TYPE)" \
  && ./configure \
    --build="$gnuArch" \
    --enable-loadable-sqlite-extensions \
    --enable-shared \
    --with-system-expat \
    --with-system-ffi \
    --without-ensurepip \
  && make -j "$(nproc)" \
  && make install \
  && ldconfig \
  \
  && find /usr/local -depth \
    \( \
      \( -type d -a \( -name test -o -name tests \) \) \
      -o \
      \( -type f -a \( -name '*.pyc' -o -name '*.pyo' \) \) \
    \) -exec rm -rf '{}' + \
  && rm -rf /usr/src/python 


# make some useful symlinks that are expected to exist
RUN cd /usr/local/bin \
  && ln -s idle3 idle \
  && ln -s pydoc3 pydoc \
  && ln -s python3 python \
  && ln -s python3-config python-config

# if this is called "PIP_VERSION", pip explodes with "ValueError: invalid truth value '<VERSION>'"
ENV PYTHON_PIP_VERSION 10.0.1

RUN set -ex; \
  \
  wget -O get-pip.py 'https://bootstrap.pypa.io/get-pip.py'; \
  \
  python get-pip.py \
    --disable-pip-version-check \
    --no-cache-dir \
    "pip==$PYTHON_PIP_VERSION" \
  ; \
  pip --version; \
  \
  find /usr/local -depth \
    \( \
      \( -type d -a \( -name test -o -name tests \) \) \
      -o \
      \( -type f -a \( -name '*.pyc' -o -name '*.pyo' \) \) \
    \) -exec rm -rf '{}' +; \
  rm -f get-pip.py




run apt-get update && apt-get install -y git && apt-get install -y python3-pip python3-dev build-essential
run pip3 install --upgrade pip
run pip3 install --upgrade virtualenv 

WORKDIR /stage/
RUN git clone https://github.com/allenai/allennlp.git && \
    pip3 install -r ./allennlp/requirements.txt && \
    rm ~/.cache/pip/ -rf

env PYTHONPATH="/stage/allennlp/:${PYTHONPATH}"
copy elmo_lstm.py /stage/allennlp/allennlp/modules/
run pip3 install pandas
run python3 -m spacy download en
run apt-get install -y vim





