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

# Set working directory
WORKDIR /repository

# Run the logging server and keep the container running
CMD ["/bin/bash", "-c", "source activate agentomics-env && tail -f /dev/null"]