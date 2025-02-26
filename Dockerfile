# Use a Python base image
FROM continuumio/miniconda3:latest

# Install sudo for later creation of the agent user
RUN apt-get update && apt-get install -y sudo

RUN echo 'root:1234' | chpasswd

# Create a non-root user for security
#RUN useradd -m -s /bin/bash appuser

# Give appuser sudo rights for specific commands without password
#RUN echo "appuser ALL=(ALL) NOPASSWD: /usr/sbin/useradd, /usr/sbin/userdel, /bin/chown, /bin/chmod" >> /etc/sudoers

# Create a directory for writable operations
RUN mkdir /workspace 
#&& \ chown appuser:appuser /workspace

# Create runs directory
RUN mkdir /workspace/runs

# Set working directory
WORKDIR /repository

# Copy environment.yaml file
COPY environment.yaml .

# Create conda environment
RUN conda env create -f environment.yaml

# Give user ownership of conda environment directory
#RUN chown -R appuser:appuser /opt/conda

# Copy datasets into the agent workspace
COPY datasets/*/(?!.*test).* /workspace/datasets/

# Switch to non-root user
#USER appuser

# Run the logging server and keep the container running
CMD ["/bin/bash", "-c", "source activate multiagent-ml-env && tail -f /dev/null"]