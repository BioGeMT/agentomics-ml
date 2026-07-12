FROM condaforge/mambaforge:24.9.2-0

ARG REPOSITORY_BRANCH=main
RUN git clone --depth 1 --branch "${REPOSITORY_BRANCH}" \
    https://github.com/BioGeMT/Agentomics-ML.git /repository

# Always set -y to conda install commands
ENV CONDA_ALWAYS_YES=true 
# Cache conda packages in a temp directory (removed after build - reduces image size)
ENV CONDA_PKGS_DIRS=/tmp/conda-pkgs
# Similar as above but for pip
ENV PIP_NO_CACHE_DIR=1
# Suppress pip version warnings
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Create the conda environment with mamba for speed and memory efficiency.
RUN mamba env create -f /repository/envs/environment.yaml \
    && mamba clean -afy \
    && rm -rf /tmp/conda-pkgs

# Initialize conda for bash and set up auto-activation
RUN conda init bash \
    && echo "conda activate agentomics-env" >> /root/.bashrc

# Setup agent start environment
ENV START_ENV_PKG=/opt/agent_start_env.tar.gz
# Preload libgomp on env activation so it claims a static TLS slot before
# torch's shared libs consume them all. Without this, sklearn (imported via
# transformers) fails with "cannot allocate memory in static TLS block".
RUN mamba env create -f /repository/envs/environment_agent.yaml \
    && mkdir -p /opt/conda/envs/agent_start_env/etc/conda/activate.d \
    && echo 'export LD_PRELOAD=$CONDA_PREFIX/lib/libgomp.so.1' \
       > /opt/conda/envs/agent_start_env/etc/conda/activate.d/preload_libgomp.sh \
    && mamba clean -afy \
    && rm -rf /tmp/conda-pkgs \
    && conda run -n agent_start_env conda-pack -o ${START_ENV_PKG} \
    && conda env remove -n agent_start_env

# Create a restricted user for sandboxed agent tool execution.
# The runtime (root) owns the workspace; only current_step_dir is chowned to this user per step.
RUN useradd -m -s /bin/bash agentomics-agent
ENV AGENT_USER=agentomics-agent
ENV AGENTOMICS_WORKSPACE_DIR=/workspace
RUN mkdir -p ${AGENTOMICS_WORKSPACE_DIR}

WORKDIR /repository

ENTRYPOINT ["/repository/run.sh"]
