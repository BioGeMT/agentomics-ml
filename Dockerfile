FROM condaforge/mambaforge:23.3.1-0

# Always set -y to conda install commands
ENV CONDA_ALWAYS_YES=true 
# Cache conda packages in a temp directory (removed after build - reduces image size)
ENV CONDA_PKGS_DIRS=/tmp/conda-pkgs
# Similar as above but for pip
ENV PIP_NO_CACHE_DIR=1
# Suppress pip version warnings
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && rm -rf /var/lib/apt/lists/*

# Copy & create your conda environment\ using environment.yaml (with mamba for speed and memory efficiency)
COPY environment.yaml .
RUN mamba env create -f environment.yaml \
    && mamba clean -afy \
    && rm -rf /tmp/conda-pkgs

# Initialize conda for bash and set up auto-activation
RUN conda init bash \
    && echo "conda activate agentomics-env" >> /root/.bashrc

# Pre-download foundation models
RUN mkdir -p /foundation_models /cache/foundation_models
ENV HF_HOME=/cache/foundation_models
COPY foundation_models/ /foundation_models/
COPY src/utils/foundation_models_utils.py /repository/src/utils/foundation_models_utils.py
COPY src/utils/download_foundation_models.py /repository/src/utils/download_foundation_models.py
RUN LD_PRELOAD=$(find /opt/conda/envs/agentomics-env/lib/python3.12/site-packages/scikit_learn.libs -name "libgomp*.so*" | head -1) \
    /opt/conda/envs/agentomics-env/bin/python /repository/src/utils/download_foundation_models.py

# Setup agent start environment
ENV START_ENV_PKG=/opt/agent_start_env.tar.gz
COPY environment_agent.yaml .
RUN mamba env create -f environment_agent.yaml \
    && mamba clean -afy \
    && rm -rf /tmp/conda-pkgs
RUN conda run -n agent_start_env conda-pack -o ${START_ENV_PKG}

WORKDIR /repository

ENTRYPOINT ["/opt/conda/envs/agentomics-env/bin/python", "/repository/src/run_agent_interactive.py"]