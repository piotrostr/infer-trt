FROM nvcr.io/nvidia/tensorrt:22.03-py3

RUN apt-get update && apt-get install -y wget

RUN wget --quiet \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  -O ~/miniconda.sh \
  && /bin/bash ~/miniconda.sh -b -p /opt/conda

RUN conda install -y -c conda-forge opencv
