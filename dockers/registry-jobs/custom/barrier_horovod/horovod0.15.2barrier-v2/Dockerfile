FROM phillyregistry.azurecr.io/philly/jobs/toolkit/tensorflow:community-gpu-1.12.0-cp35

# Labels for the docker
LABEL description="Horovod Barrier V2" \
      repository="philly/jobs/custom/barrier_horovod" \
      tag="Horovod-v0.15.2-Barrier-v2" \
      creator="yuanya" \
      tooltype="tensorflow" \
      tooltypeversion="1.12" \
      createtime="14/06/2019"

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64
ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu

ENV PATH=$PATH:/usr/local/cuda-9.0/bin:/usr/local/cuda9/bin:/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib/stubs

RUN ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcublas.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcublas.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so.1 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcufft.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcufft.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcufftw.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcufftw.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcurand.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcurand.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcusolver.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcusolver.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcusparse.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcusparse.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppc.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppc.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppial.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppial.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppicc.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppicc.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppicom.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppicom.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppidei.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppidei.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppif.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppif.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppig.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppig.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppim.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppim.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppist.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppist.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppisu.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppisu.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppitc.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnppitc.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnpps.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnpps.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnvgraph.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnvgraph.so.9.0 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnvidia-ml.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnvidia-ml.so.1 && \ 
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnvrtc.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnvrtc.so.9.0

# update pip
# RUN pip install --upgrade pip && pip install --upgrade setuptools
RUN pip3 --default-timeout=10000 install --upgrade pip

# install pytorh
 RUN pip3 install --no-cache-dir torch==1.0.0

# install tensorflow_ranking
RUN pip3 --default-timeout=10000 install tensorflow_ranking==0.1.2

# install horovod
RUN mkdir /tmp/Horovod && \
    cd /tmp/Horovod && \
    git clone --recursive https://github.com/amazingyyc/horovod.git && \
    cd horovod && \
    git checkout -b v0.15.2barrier origin/v0.15.2barrier && \
    python3 setup.py sdist && \ 
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=0 pip3 --default-timeout=10000 install dist/horovod-0.15.2.tar.gz && \
    ldconfig

RUN rm -rf /home/job
RUN mkdir /home/job
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute
WORKDIR /home/job/