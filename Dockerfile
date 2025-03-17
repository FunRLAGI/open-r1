# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-11.html#rel-23-11
FROM docker.1ms.run/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
LABEL maintainer="open-r1"

# ARG DEBIAN_FRONTEND=noninteractive

# ARG PYTORCH='2.5.1'
# # Example: `cu102`, `cu113`, etc.
# ARG CUDA='cu124'

RUN apt-get update && apt-get install -y \
    wget \
    curl \
    libaio-dev \
    git \
    build-essential

#install uv 

RUN  pip install uv


RUN  git clone https://gh-proxy.com/github.com/FunRLAGI/open-r1.git  

SHELL ["/bin/bash", "-c"]

RUN uv venv openr1 --python 3.11 

RUN source openr1/bin/activate 


RUN uv pip install --system --upgrade pip

RUN uv pip install --system  setuptools && uv pip install --system flash-attn --no-build-isolation

RUN uv pip install vllm==0.7.2 --system



RUN cd open-r1&& uv pip install --system -r requirements.txt  

RUN cd open-r1 && GIT_LFS_SKIP_SMUDGE=1 uv pip install --system -e ".[dev]" 
