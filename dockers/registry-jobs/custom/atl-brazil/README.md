# ATL Brazil custom dockers

This folder contains the custom dockers used by the Microsoft Advanced Technology Labs - Brazil (ATL-Brazil).

For now, we are using the latest version of CNTK, but we have plans to use our own CNTK wheel.

## latest
See ##cntk2.5-py3.5.

## cntk2.5-py3.5

* Base image: `philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7`
* Python virtual environment: [Miniconda](https://conda.io/miniconda.html) 4.4.10
* Toolkit: CNTK 2.5.1 (with MKLML 2018.0.1.20171227 and MKLDNN v0.12)
* Main environment: `cntk-py35`. Packages installed:
    - Python=3.5
    - Pandas
    - h5py
    - numpy
    - scipy
    - scikit-learn
    - opencv=3.1 (from conda-forge channel)
    - easydict (pip)
    - azure (pip)

## Contributors
- Igor Macedo Quintanilha <v-igquin@microsoft.com> @ ATL-Brazil
- Roberto de Moura Estevão Filho <v-rodemo@microsoft.com> @ ATL-Brazil