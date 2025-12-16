# ingestion.py
import pandas as pd
from sqlalchemy import create_engine, text
from config import DATABASE_URL
import os
import numpy as np

data_path = 'data/household_power_consumption.txt'

# expected columns before parsing date
expected_raw_cols = ['Date', 'Time', 'Global_active_power', 'Global_reactive_power',
                     'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

# Helper: reduce memory usage of numeric columns
def downcast_df(df):
    for col in df.select_dtypes(include=['float64','int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Create table if not exists (schema)
create_table_query = """
CREATE TABLE IF NOT EXISTS power_consumption (
    Datetime TIMESTAMP,
    Global_active_power FLOAT,
    Global_reactive_power FLOAT,
    Voltage FLOAT,
    Global_intensity FLOAT,
    Sub_metering_1 FLOAT,
    Sub_metering_2 FLOAT,
    Sub_metering_3 FLOAT
);
"""
with engine.begin() as conn:
    conn.execute(text(create_table_query))

# Read + ingest in chunks
chunksize = 100000   # tune this smaller if you still hit memory issues (e.g., 20000)
total_rows = 0
chunk_no = 0

# We'll read Date and Time as strings then combine to datetime per chunk
reader = pd.read_csv(
    data_path,
    sep=';',
    usecols=expected_raw_cols,
    dtype={
        'Date': 'string',
        'Time': 'string',
        'Global_active_power': 'float64',
        'Global_reactive_power': 'float64',
        'Voltage': 'float64',
        'Global_intensity': 'float64',
        'Sub_metering_1': 'float64',
        'Sub_metering_2': 'float64',
        'Sub_metering_3': 'float64'
    },
    na_values=['?', ''],
    iterator=True,
    chunksize=chunksize,
    low_memory=True
)

for chunk in reader:
    chunk_no += 1
    chunk['Datetime'] = pd.to_datetime(chunk['Date'].str.strip() + ' ' + chunk['Time'].str.strip(),
                                       dayfirst=True, errors='coerce')

    # Drop the raw Date/Time columns
    chunk.drop(columns=['Date','Time'], inplace=True)

    # Remove rows where all power-related fields are NaN
    power_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']
    chunk = chunk.dropna(subset=power_cols, how='all')

    # Schema check: ensure required columns exist
    expected_cols = ['Datetime', 'Global_active_power', 'Global_reactive_power',
                     'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    missing = [c for c in expected_cols if c not in chunk.columns]
    if missing:
        raise ValueError(f"Missing columns in chunk: {missing}")

    # Downcast numeric dtypes to save memory
    chunk = downcast_df(chunk)

    # Optionally reorder columns to match DB
    chunk = chunk[expected_cols]

    # Ingest chunk to DB
    with engine.begin() as conn:
        chunk.to_sql('power_consumption', conn, if_exists='append', index=False, method='multi')
    rows = len(chunk)
    total_rows += rows
    print(f"Chunk {chunk_no} ingested, rows={rows}, total={total_rows}")

print(f"Completed ingestion. Total rows ingested: {total_rows}")
