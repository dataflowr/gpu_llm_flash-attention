#!/bin/bash

# This init script clones a Git repository that contains a Jupyter notebook
# named `FlashAttention_empty.ipynb` and opens it in Jupyter Lab at startup
# Expected parameters : None

# Clone repository and give permissions to the onyxia user
GIT_REPO=gpu_llm_flash-attention
git clone --depth 1 https://github.com/dataflowr/${GIT_REPO}.git
chown -R onyxia:users ${GIT_REPO}/

# Install the package in editable mode
pip install -e ${GIT_REPO}

# Open the relevant notebook when starting Jupyter Lab
echo "c.LabApp.default_url = '/lab/tree/${GIT_REPO}/FlashAttention_empty.ipynb'" >> /home/onyxia/.jupyter/jupyter_server_config.py