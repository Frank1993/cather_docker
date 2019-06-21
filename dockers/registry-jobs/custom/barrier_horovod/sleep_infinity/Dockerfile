FROM amazingyyc/horovod_barrier_all_gather_str:latest

# Labels for the docker
LABEL description="sleep_infinity" \
      repository="philly/jobs/custom/barrier_horovod" \
      tag="sleep_infinity" \
      creator="yuanya" \
      tooltype="tensorflow" \
      tooltypeversion="1.12" \
      createtime="20/06/2019"

#install tensorflow_ranking
RUN pip3 --default-timeout=10000 install --upgrade pip
RUN pip3 --default-timeout=10000 install tensorflow_ranking==0.1.2

RUN mkdir /home/job
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute
WORKDIR /home/job/