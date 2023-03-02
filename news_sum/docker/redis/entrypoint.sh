#!/usr/bin/dumb-init /bin/sh

### docker entrypoint script, for starting redis stack server
BASEDIR=/opt/redis-stack
cd ${BASEDIR}

CMD=${BASEDIR}/bin/redis-stack-server
if [ -f /redis-stack.conf ]; then
    CONFFILE=/redis-stack.conf
fi

if [ -z "${REDIS_DATA_DIR}" ]; then
    REDIS_DATA_DIR=/data
fi

${CMD} \
--dir ${REDIS_DATA_DIR} \
${REDIS_ARGS}
