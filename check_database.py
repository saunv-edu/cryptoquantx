# check_database.py
import sqlite3
import pandas as pd

def check_database():
    conn = sqlite3.connect('trading.db')
    
    # Check tables
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print("📋 Database tables:", tables)
    
    # Check klines data
    try:
        df = pd.read_sql_query("SELECT * FROM klines LIMIT 5", conn)
        print(f"\n📊 Sample klines data ({len(df)} rows):")
        print(df)
    except Exception as e:
        print(f"No klines data yet: {e}")
    
    conn.close()

if __name__ == "__main__":
    check_database()