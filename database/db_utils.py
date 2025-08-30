import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'database', 'planning_results.db')

def create_tables():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS shipment_plan (
            sku TEXT,
            dc TEXT,
            week INTEGER,
            demand INTEGER,
            allocated INTEGER,
            total_trucks INTEGER,
            safety_met BOOLEAN
        )
    ''')
    conn.commit()
    conn.close()

def save_shipment_plan(df):
    conn = sqlite3.connect(DB_PATH)
    df.to_sql('shipment_plan', conn, if_exists='replace', index=False)
    conn.close()
