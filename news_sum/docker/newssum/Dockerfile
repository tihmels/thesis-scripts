FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends  \
    build-essential \
    git \
    wget \
    python3-opencv \
    ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home

RUN mkdir s3d && cd s3d  \
    && wget -O s3d_howto100m.pth https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_howto100m.pth  \
    && wget -O dict.npy https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_dict.npy

ARG VERSION=14
RUN git clone https://github.com/tihmels/thesis-scripts.git

COPY requirements.txt .
RUN pip3 install -r requirements.txt --no-cache-dir
RUN apt-get update && apt-get install -y --no-install-recommends dumb-init && rm -rf /var/lib/apt/lists/*

RUN mkdir out && mkdir TS

WORKDIR /home/thesis-scripts/news_sum/
