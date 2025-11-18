NUM_FACTORS = 64
TOP_K_RECO = 100

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'db': 'shop',
}

FETCH_SQL = """
    SELECT user, item, times AS rating 
    FROM suki;
    """

OUTPUT_DIR = "output"

DB_URI = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['db']}"