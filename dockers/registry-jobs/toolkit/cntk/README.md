# CNTK images for Philly

The CNTK images are based out of the released versions of CNTK.
Differences from the released docker image in Dockerhub:
1) We use the OpenMPI with OFED drivers corresponding to what we have in our clusters
2) NCCL - to allow OpenMpi to use NCCL verbs where it can (on-prem and AP).
3) Dev tools

2.5-private-build-python3.5:
This image is used for using nightly and private builds from the CNTK 2.5 build pipeline.
The toolkit-execute supports:
--buildID - build id from the nightly build (starting from 19448).
--flavor - default is release.
--buildType - default is gpu.

Owners/Contributors: vivram, paverma, juqia

Please contact the owners for any updates, changes or fixes.