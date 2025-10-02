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