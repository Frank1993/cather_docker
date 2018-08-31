#!/bin/bash
#This script is used for printing out a warning message and executing the commands supplied in the docker run.
#It's because how ENTRYPOINT works in docker, we need this extra step otherwise docker run will quit immediately after echo
/bin/echo "********WARNING******** \n This docker image does not fully support using InfinityBand across multiple containers yet. Please disable IB when running distributed learning."
"$@"