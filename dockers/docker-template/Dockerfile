FROM private-custom-scale-image

# Create the home directory for the custom-docker
RUN mkdir /home/custom

# Copy the files to the necessary folder
COPY runCust /home/custom

# Make sure the permissions are correct
RUN chmod u+x /home/custom/runCust

# Everything above this line is required for the docker.
# Add your personalized features below here.