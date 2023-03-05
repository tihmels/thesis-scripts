FROM ubuntu:20.04

RUN apt-get update &&  \
    apt-get install -y --no-install-recommends sudo build-essential python3 python3-dev python3-pip wget unzip git &&  \
    rm -rf /var/lib/apt/lists/*

WORKDIR /home

RUN python3 -m pip install --ignore-installed regex &&  \
    git clone --recursive https://github.com/RedisAI/RedisAI &&  \
    cd RedisAI && ./get_deps.sh && make setup -C opt && make -C opt clean ALL=1 && make -C opt

FROM redis/redis-stack-server
COPY --from=0 /home/RedisAI/bin/linux-x64-release/install-cpu/redisai.so /opt/redis-stack/lib/redisai.so

ENTRYPOINT ["/opt/redis-stack/bin/redis-stack-server"]



