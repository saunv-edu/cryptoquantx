import sqlite3
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import logger

def fix_database_schema():
    """Fix database schema by adding missing columns."""
    conn = sqlite3.connect('trading.db')
    cursor = conn.cursor()
    
    try:
        # Check current columns in indicators table
        cursor.execute("PRAGMA table_info(indicators)")
        current_columns = [col[1] for col in cursor.fetchall()]
        
        print("Current columns:", current_columns)
        
        # Columns to add
        columns_to_add = [
            'stoch_k REAL DEFAULT 0',
            'stoch_d REAL DEFAULT 0', 
            'sma_50 REAL DEFAULT 0',
            'bb_width REAL DEFAULT 0',
            'bb_position REAL DEFAULT 0',
            'volume_sma_20 REAL DEFAULT 0',
            'obv REAL DEFAULT 0'
        ]
        
        for column_def in columns_to_add:
            column_name = column_def.split(' ')[0]
            if column_name not in current_columns:
                try:
                    cursor.execute(f"ALTER TABLE indicators ADD COLUMN {column_def}")
                    print(f"✅ Added column: {column_name}")
                except Exception as e:
                    print(f"❌ Failed to add {column_name}: {e}")
            else:
                print(f"✅ Column already exists: {column_name}")
        
        conn.commit()
        print("🎉 Database schema fixed successfully!")
        
    except Exception as e:
        conn.rollback()
        print(f"💥 Error fixing database: {e}")
        raise
    finally:
        conn.close()

def create_fresh_database():
    """Create a fresh database with correct schema."""
    if os.path.exists('trading.db'):
        os.rename('trading.db', 'trading.db.backup')
        print("📁 Backed up existing database")
    
    # Import db manager to create new tables
    from data.databases.db_manager import DatabaseManager
    db_manager = DatabaseManager()
    print("🆕 Created fresh database with latest schema")

if __name__ == "__main__":
    print("🔧 Database Fix Tool")
    print("1. Fix existing database")
    print("2. Create fresh database")
    
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice == "1":
        fix_database_schema()
    elif choice == "2":
        create_fresh_database()
    else:
        print("Invalid choice")