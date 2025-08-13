FROM condaforge/mambaforge:23.3.1-0

# Set memory-efficient conda/mamba settings
ENV CONDA_ALWAYS_YES=true
ENV CONDA_PKGS_DIRS=/tmp/conda-pkgs
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install bc calculator for floating point arithmetic
RUN apt-get update && apt-get install -y bc && rm -rf /var/lib/apt/lists/*

# Create a non-root user "agent" and give it a home
RUN useradd -m -s /bin/bash agent \
    && echo 'agent:changeme' | chpasswd

# Prepare workspace
RUN mkdir -p /workspace/runs

# Copy & create your conda environment using environment.yaml (with mamba for speed and memory efficiency)
COPY environment.yaml .
RUN mamba env create -f environment.yaml \
    && mamba clean -afy \
    && rm -rf /tmp/conda-pkgs

# Initialize conda for bash and set up auto-activation
RUN conda init bash \
    && echo "conda activate agentomics-env" >> /home/agent/.bashrc \
    && echo "conda activate agentomics-env" >> /root/.bashrc

# Copy the shared entrypoint script
COPY agentomics-entrypoint.py /usr/local/bin/agentomics-entrypoint.py
RUN chmod +x /usr/local/bin/agentomics-entrypoint.py

WORKDIR /repository

# Set the entrypoint to use our shared script
# ENTRYPOINT ["python", "/usr/local/bin/agentomics-entrypoint.py"]