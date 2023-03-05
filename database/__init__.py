import os

from smartredis import Client
from walrus import *

host = os.getenv('REDIS_HOST', default='localhost')
port = os.getenv('REDIS_PORT', default=6379)

print(f'RedisConn: {host}:{port}')

db = Database(host=host, port=port, db=0, decode_responses=True)
rai = Client(address=f'{host}:{port}')
