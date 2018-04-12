
=================================
General Dockers for Philly-Tools.
=================================

These are "all-dressed" dockers with all the main deep learning tools.
They come in two versions: python 2.7, python 3.6.

The python 3.6 version contains:
    - pytorch 0.3.1
    - torchvision-0.2.0 
    - tensorflow 1.7
    - theano 0.9
    - Lasagne 0.2
    - Chainer 3.5
    - cupy 2.5

In each docker, model directory and data directories are stored in environment variables
PT_OUTPUT_DIR and PT_DATA_DIR.

Users can install their run-time dependencies in various ways:
- by adding a requirements.txt in the folder which contains their configFile (script to launch)
- by adding a setup.sh in the folder which contains their configFile

Python is ran with the -B flag in order not to create the __pycache__ folder.