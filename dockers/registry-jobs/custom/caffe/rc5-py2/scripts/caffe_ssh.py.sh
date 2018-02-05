#!/bin/bash

set -x
cwd=$(dirname "${BASH_SOURCE[0]}")
newCaffe="$cwd/caffe.tar.gz"

if [[ -f "$newCaffe" ]]; then
    sudo tar --strip-components=2 -C /opt/ -xvf $newCaffe
fi

if [[ -f "$cwd"/bash_history ]]; then
    cp $cwd/bash_history ~/.bash_history
fi

#sudo apt-get install git tmux
#sudo pip install pyyaml progressbar easydict ete2
#sudo pip install nltk
#python -m nltk.downloader all
