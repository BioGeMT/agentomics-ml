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

# Copy & create your conda environment using environment.yaml (with mamba for speed and memory efficiency)
COPY envs/environment.yaml .
RUN mamba env create -f environment.yaml \
    && mamba clean -afy \
    && rm -rf /tmp/conda-pkgs

# Initialize conda for bash and set up auto-activation
RUN conda init bash \
    && echo "conda activate agentomics-env" >> /root/.bashrc

# Pre-download foundation models
RUN mkdir -p /foundation_models /cache/foundation_models
ENV HF_HOME=/cache/foundation_models
ARG FOUNDATION_MODELS_TYPE=
COPY foundation_models/ /foundation_models/
COPY src/utils/foundation_models_utils.py /repository/src/utils/foundation_models_utils.py
COPY src/utils/download_foundation_models.py /repository/src/utils/download_foundation_models.py
RUN if [ -n "$FOUNDATION_MODELS_TYPE" ]; then \
      export LD_PRELOAD=$(find /opt/conda/envs/agentomics-env -path "*/scikit_learn.libs/libgomp*.so*" | head -1) && \
      /opt/conda/envs/agentomics-env/bin/python /repository/src/utils/download_foundation_models.py \
        --foundation-models-type "$FOUNDATION_MODELS_TYPE" \
        --models-yaml /foundation_models/models.yaml; \
    else \
      echo "Skipping foundation model download (FOUNDATION_MODELS_TYPE not set)"; \
    fi

# Setup agent start environment
ENV START_ENV_PKG=/opt/agent_start_env.tar.gz
COPY envs/environment_agent.yaml .
RUN mamba env create -f environment_agent.yaml \
    && mamba clean -afy \
    && rm -rf /tmp/conda-pkgs
RUN conda run -n agent_start_env conda-pack -o ${START_ENV_PKG}

# Create a restricted user for sandboxed agent tool execution.
# The runtime (root) owns the workspace; only current_step_dir is chowned to this user per step.
# /cache/foundation_models is handed to this user so the agent can download new HF models at runtime.
RUN useradd -m -s /bin/bash agentomics-agent \
    && chown -R agentomics-agent /cache/foundation_models
ENV AGENT_USER=agentomics-agent

WORKDIR /repository

ENTRYPOINT ["/opt/conda/envs/agentomics-env/bin/python", "/repository/src/run_agent_interactive.py"]
