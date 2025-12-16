# db_test.py
from sqlalchemy import create_engine, text
from config import DATABASE_URL
engine = create_engine(DATABASE_URL)
with engine.connect() as conn:
    r = conn.execute(text("SELECT 1")).scalar()
    print("DB test query returned:", r)
