#!/usr/bin/dumb-init /bin/sh

BASEDIR=/home/thesis-scripts/news_sum/
cd ${BASEDIR}

CMD=${BASEDIR}/bin/redis-stack-server

if [ -z "${REDIS_DATA_DIR}" ]; then
    REDIS_DATA_DIR=/data
fi

${CMD} --dir ${REDIS_DATA_DIR}
