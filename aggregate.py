# aggregate_daily.py
"""
Aggregate minute-level household power consumption raw table into daily table.

Requirements:
- config.py with DATABASE_URL (SQLAlchemy connection string)
- Raw table name: power_consumption (has Datetime column)
Output:
- MySQL table: daily_power_consumption (replaced)
- CSV: processed_data/daily_power_consumption.csv

This version converts daily sums for Global_active_power and Global_reactive_power
from "sum of per-minute kW samples" into kWh/day by dividing those sums by 60.
"""

import os
from sqlalchemy import create_engine, text
import pandas as pd
from config import DATABASE_URL

RAW_TABLE = "power_consumption"
DAILY_TABLE = "daily_power_consumption"
DATETIME_COL = "Datetime"
OUTPUT_DIR = "processed_data"
CSV_OUT = os.path.join(OUTPUT_DIR, "daily_power_consumption.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# sensible default aggregations
SUM_COLS = [
    "Global_active_power",
    "Global_reactive_power",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]
MEAN_COLS = [
    "Voltage",
]

def get_table_columns(engine, db_name, table_name):
    """
    Returns a list of column names present in the table (preserves case).
    """
    sql = f"""
    SELECT COLUMN_NAME
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = :db AND TABLE_NAME = :table
    ORDER BY ORDINAL_POSITION;
    """
    with engine.connect() as conn:
        result = conn.execute(text(sql), {"db": db_name, "table": table_name})
        cols = [row[0] for row in result.fetchall()]
    return cols

def build_agg_sql(db_name, raw_table, datetime_col, sum_cols, mean_cols, present_cols):
    """
    Build SQL string to aggregate per DATE(datetime_col).
    Only include aggregations for columns that actually exist.
    """
    agg_parts = []
    for c in sum_cols:
        if c in present_cols:
            agg_parts.append(f"SUM(`{c}`) AS `{c}_sum`")
    for c in mean_cols:
        if c in present_cols:
            agg_parts.append(f"AVG(`{c}`) AS `{c}_mean`")

    if not agg_parts:
        raise ValueError("No recognized numeric columns found to aggregate.")

    # Use DATE() to get day; keep DAY as 'day' column (date)
    agg_sql = f"""
    SELECT DATE(`{datetime_col}`) AS `day`, {', '.join(agg_parts)}
    FROM `{raw_table}`
    GROUP BY DATE(`{datetime_col}`)
    ORDER BY DATE(`{datetime_col}`);
    """
    return agg_sql

def main():
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    # Extract DB name from DATABASE_URL (simple heuristic)
    # DATABASE_URL examples: mysql+pymysql://user:pass@host:3306/dbname
    db_name = None
    try:
        # parse last component
        db_name = DATABASE_URL.rsplit("/", 1)[-1]
        # strip params if any
        db_name = db_name.split("?")[0]
    except Exception:
        pass

    if not db_name:
        print("Warning: couldn't infer DB name from DATABASE_URL; information_schema query may fail.")
        db_name = engine.url.database

    # check present columns
    present_cols = get_table_columns(engine, db_name, RAW_TABLE)
    if not present_cols:
        raise RuntimeError(f"Table `{RAW_TABLE}` not found or empty in DB `{db_name}`.")

    print("Detected columns in raw table:", present_cols)

    # Build SQL aggregation dynamically for columns present
    agg_sql = build_agg_sql(db_name, RAW_TABLE, DATETIME_COL, SUM_COLS, MEAN_COLS, present_cols)
    print("Aggregation SQL built. Running query...")

    # Run aggregation
    df_daily = pd.read_sql(agg_sql, engine, parse_dates=["day"])
    if df_daily.empty:
        print("Aggregation returned zero rows - check raw table content.")
        return

    # rename 'day' to Datetime to be consistent
    df_daily = df_daily.rename(columns={"day": "Datetime"})
    # optional: convert Datetime to date (no time)
    df_daily["Datetime"] = pd.to_datetime(df_daily["Datetime"]).dt.date

    # --- CONVERSION: convert sums of per-minute kW samples to kWh/day ---
    # if columns exist, divide by 60 to convert (kW * minutes -> kWh)
    if "Global_active_power_sum" in df_daily.columns:
        df_daily["Global_active_power_sum"] = df_daily["Global_active_power_sum"].astype(float) / 60.0
    if "Global_reactive_power_sum" in df_daily.columns:
        df_daily["Global_reactive_power_sum"] = df_daily["Global_reactive_power_sum"].astype(float) / 60.0

    # Optionally, rename converted columns for clarity (uncomment if desired)
    # df_daily = df_daily.rename(columns={
    #     "Global_active_power_sum": "Global_active_power_kwh",
    #     "Global_reactive_power_sum": "Global_reactive_power_kwh"
    # })

    # Save to CSV
    df_daily.to_csv(CSV_OUT, index=False)
    print(f"Saved aggregated daily CSV to {CSV_OUT} (rows: {len(df_daily)})")

    # Push to MySQL (replace if exists)
    print(f"Writing daily table to MySQL as `{DAILY_TABLE}` (replace if exists)...")
    # convert Datetime to proper datetime (midnight) before to_sql
    df_mysql = df_daily.copy()
    df_mysql["Datetime"] = pd.to_datetime(df_mysql["Datetime"])
    df_mysql.to_sql(DAILY_TABLE, engine, if_exists="replace", index=False, method="multi", chunksize=5000)
    print(f"Daily table `{DAILY_TABLE}` written to DB. Done. Rows written: {len(df_mysql)}")

    # Quick head preview
    print(df_daily.head())

if __name__ == "__main__":
    main()
