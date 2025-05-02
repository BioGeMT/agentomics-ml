# Use a Python base image
FROM continuumio/miniconda3:latest

# Install sudo for later creation of the agent user
RUN apt-get update && apt-get install -y sudo

# Set the root password
RUN echo 'root:1234' | chpasswd

# Create a directory for writable operations
RUN mkdir /workspace 

# Create runs directory
RUN mkdir /workspace/runs

# Copy environment.yaml file
COPY environment.yaml .

# Create conda environment
RUN conda env create -f environment.yaml

# Copy datasets into the agent workspace
COPY datasets/ /workspace/datasets/

# Delete all files containing "test" in the name
RUN find /workspace/datasets -type f -name "*test*" -delete

# Delete all files containing "metadata" in the name
RUN find /workspace/datasets -type f -name "*metadata*" -delete

# Make all the files read only for everyone
RUN chmod -R o-w /workspace/datasets

# Make datasets folder accessible
RUN chmod -R o+x /workspace/datasets

# Set working directory
WORKDIR /repository

# Run the logging server and keep the container running
CMD ["/bin/bash", "-c", "source activate agentomics-env && tail -f /dev/null"]