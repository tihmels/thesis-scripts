import redis

# red = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)


red = redis.Redis(host='redis-13975.c91.us-east-1-3.ec2.cloud.redislabs.com',
                  port=13975,
                  db=0,
                  decode_responses=True,
                  username='default',
                  password='YX0Nx3ddpclPyewTGzvswBnZrPyT9Tit')
