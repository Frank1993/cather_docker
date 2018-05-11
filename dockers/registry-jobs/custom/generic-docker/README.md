
=================================
General Dockers for Philly-Tools.
=================================

These are "all-dressed" dockers with all the main deep learning tools.
They come in four versions:
    - python 2.7 and python 3.6 with cuda 8 (only for chainer)
    - python 2.7 and python 3.6 with cuda 9.

The py27 and py36 versions contain:
    - pytorch 0.3.1
    - torchvision-0.2.0
    - tensorflow 1.7
    - theano 0.9
    - Lasagne 0.2
    - Chainer 3.5
    - cupy-cuda90
    - keras

The cuda8-py27 dockers contain:
    - Chainer 1.24
    - cupy-cuda80

The cuda8-py36 docker contain:
    - Chainer 5.0
    - cupy-cuda80


In each docker, model directory and data directories are stored in environment variables
PT_OUTPUT_DIR and PT_DATA_DIR.

Users can install their run-time dependencies in various ways:
- by adding a requirements.txt in the folder which contains their configFile (script to launch)
- by adding a setup.sh in the folder which contains their configFile

Python is ran with the -B flag in order not to create the __pycache__ folders.