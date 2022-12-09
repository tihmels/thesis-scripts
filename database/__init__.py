from walrus import *
from smartredis import Client

host = 'localhost'
port = 6379

db = Database(host=host, port=port, db=0, decode_responses=True)
rai = Client(address=f'{host}:{port}')
