FROM phillyregistry.azurecr.io/philly/jobs/custom/pytorch:pytorch0.4.0-py36-rl-robo

# Copy the files to the necessary folder
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod u+x /home/job/toolkit-execute

# Labels for the docker
LABEL description="This docker has pytorch 0.4.1, gym[all], pytorchvision opencv in anaconda python36 with cuda 9.0, and cudnn 7.0." \
      lepository="philly/jobs/custom/pytorch" \
      tag="pytorch0.4.0-py36-rl-baseline" \
      creator="v-yuewng" tooltype="pytorch" \
      tooltypeversion="0.4.0" \
      createtime="3/6/2019"
# Everything above this line is required for the docker.
# Add your personalized features below here.
 

RUN pip install pandas tensorflow-gpu


RUN git clone https://github.com/openai/baselines.git \
&& cd baselines \
&& pip install -e .

WORKDIR /workspace
RUN chmod -R a+w /workspace \
&& chmod -R 777 /opt/conda/

 
