FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends  \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home

RUN mkdir s3d && cd s3d  \
    && wget -O s3d_howto100m.pth https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_howto100m.pth  \
    && wget -O dict.npy https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_dict.npy

RUN git clone https://github.com/tihmels/thesis-scripts.git

COPY requirements.txt .
RUN pip3 install -r requirements.txt --no-cache-dir

RUN mkdir out && mkdir TS

