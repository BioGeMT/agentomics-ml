FROM condaforge/mambaforge:23.3.1-0

# Always set -y to conda install commands
ENV CONDA_ALWAYS_YES=true 
# Cache conda packages in a temp directory (removed after build - reduces image size)
ENV CONDA_PKGS_DIRS=/tmp/conda-pkgs
# Similar as above but for pip
ENV PIP_NO_CACHE_DIR=1
# Suppress pip version warnings
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install sudo for later creation of the agent user
RUN apt-get update && apt-get install -y sudo && rm -rf /var/lib/apt/lists/*

# Copy & create your conda environment\ using environment.yaml (with mamba for speed and memory efficiency)
COPY environment.yaml .
RUN mamba env create -f environment.yaml \
    && mamba clean -afy \
    && rm -rf /tmp/conda-pkgs

# Initialize conda for bash and set up auto-activation
RUN conda init bash \
    && echo "conda activate agentomics-env" >> /root/.bashrc

WORKDIR /repository

ENTRYPOINT ["/opt/conda/envs/agentomics-env/bin/python", "/repository/src/run_agent_interactive.py"]