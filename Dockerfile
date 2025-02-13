# Use a Python base image
FROM continuumio/miniconda3:latest

# Create a non-root user for security
RUN useradd -m -s /bin/bash appuser

# Create a directory for writable operations
RUN mkdir /workspace && \
    chown appuser:appuser /workspace

# Set working directory
WORKDIR /repository

# Copy environment.yaml file
COPY environment.yaml .

# Create conda environment
RUN conda env create -f environment.yaml

# Switch to non-root user
USER appuser

# Keep container running
CMD ["tail", "-f", "/dev/null"]
