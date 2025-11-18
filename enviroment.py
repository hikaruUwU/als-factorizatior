DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'db': 'shop',
}


DB_URI = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['db']}"