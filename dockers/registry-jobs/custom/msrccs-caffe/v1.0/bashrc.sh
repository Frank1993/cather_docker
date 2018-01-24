#!/bin/bash

export HISTSIZE=10000
export HISTFILESIZE=100000
shopt -s histappend

bind '"\e[A"':history-search-backward
bind '"\e[B"':history-search-forward
