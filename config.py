# config.py
from urllib.parse import quote_plus

DB_HOST = 'localhost'
DB_PORT = 3306
DB_USER = 'forecast_user'
DB_PASSWORD = 'Mahesh@1527'
DB_NAME = 'power_consumption'

# URL-encode password to safely handle special characters like @, :, etc.
DB_PASSWORD_ENC = quote_plus(DB_PASSWORD)

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD_ENC}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
