FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --recursive https://github.com/parlance/ctcdecode.git
RUN pip install ctcdecode/

RUN pip install https://github.com/kpu/kenlm/archive/master.zip

COPY ./asr.py .
COPY ./VideoData.py .
COPY ./requirements.txt .
COPY model/ ./model/
COPY ./test/story_2.wav ./test.wav

RUN pip install -r ./requirements.txt

