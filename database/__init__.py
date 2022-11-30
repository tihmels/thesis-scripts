import redis

red = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# red = redis.Redis(host=args.host, port=args.port, db=args.db, decode_responses=True, username='default', password = 'YX0Nx3ddpclPyewTGzvswBnZrPyT9Tit')
