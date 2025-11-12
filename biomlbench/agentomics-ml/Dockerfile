FROM biomlbench-env

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
RUN conda env create -f environment.yaml \
    && conda clean -afy \
    && rm -rf /tmp/conda-pkgs

# Initialize conda for bash and set up auto-activation
RUN conda init bash \
    && echo "conda activate agentomics-env" >> /root/.bashrc

# Pre-download foundation models
RUN mkdir -p /foundation_models /cache/foundation_models
ENV HF_HOME=/cache/foundation_models
COPY foundation_models/ /foundation_models/
COPY foundation_models_utils.py /repository/src/utils/foundation_models_utils.py
COPY download_foundation_models.py /repository/src/utils/download_foundation_models.py
RUN /opt/conda/envs/agentomics-env/bin/python /repository/src/utils/download_foundation_models.py

# Setup agent start environment
COPY environment_agent.yaml .
RUN conda env create -f environment_agent.yaml \
    && conda clean -afy \
    && rm -rf /tmp/conda-pkgs
RUN conda run -n agent_start_env conda-pack -o /opt/agent_start_env.tar.gz

WORKDIR /repository

# where to put submission, will be extracted
ARG SUBMISSION_DIR
ENV SUBMISSION_DIR=${SUBMISSION_DIR}
# where to put any logs, will be extracted
ARG LOGS_DIR
ENV LOGS_DIR=${LOGS_DIR}
# where to put any code, will be extracted
ARG CODE_DIR
ENV CODE_DIR=${CODE_DIR}
# where to put any other agent-specific files, will not be necessarily extracted
ARG AGENT_DIR
ENV AGENT_DIR=${AGENT_DIR}

RUN mkdir ${LOGS_DIR} ${CODE_DIR} ${AGENT_DIR}

# put all the agent files in the expected location
COPY . ${AGENT_DIR}