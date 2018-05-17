#!/bin/bash

aria2c -k 1M -x 5 -j 5 --http-accept-gzip=true --quiet https://download-cf.jetbrains.com/python/pycharm-community-2017.2.3.tar.gz
tar -zxvf pycharm-community-2017.2.3.tar.gz -C /usr/local

