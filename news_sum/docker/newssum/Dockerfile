FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/* 

WORKDIR /home
RUN git clone https://github.com/tihmels/thesis-scripts.git

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt --no-cache-dir

ENTRYPOINT ["tail", "-f", "/dev/null"]