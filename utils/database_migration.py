import sqlite3
import sys
import os
from sqlalchemy import text

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.databases.db_manager import DatabaseManager
from utils.logger import logger

def migrate_database():
    """Migrate database to latest schema."""
    logger.info("Starting database migration...")
    
    db_manager = DatabaseManager()
    
    # Check if we need to migrate
    with db_manager.get_session() as session:
        # Try to query stoch_k column to see if it exists
        try:
            # Use text() for raw SQL with SQLAlchemy
            session.execute(text("SELECT stoch_k FROM indicators LIMIT 1"))
            logger.info("Database schema is up to date.")
            return
        except Exception as e:
            if "no such column: stoch_k" in str(e):
                logger.info("Database needs migration: adding new columns...")
                migrate_indicators_table()
            else:
                logger.error(f"Error checking database schema: {e}")
                # If it's another error, try to migrate anyway
                migrate_indicators_table()

def migrate_indicators_table():
    """Add new columns to indicators table."""
    conn = sqlite3.connect('trading.db')
    cursor = conn.cursor()
    
    try:
        # Add new columns to indicators table
        new_columns = [
            'stoch_k REAL',
            'stoch_d REAL', 
            'sma_50 REAL',
            'bb_width REAL',
            'bb_position REAL',
            'volume_sma_20 REAL',
            'obv REAL'
        ]
        
        for column_def in new_columns:
            column_name = column_def.split(' ')[0]
            try:
                cursor.execute(f"ALTER TABLE indicators ADD COLUMN {column_def}")
                logger.info(f"Added column: {column_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e):
                    logger.info(f"Column already exists: {column_name}")
                else:
                    logger.warning(f"Could not add column {column_name}: {e}")
        
        conn.commit()
        logger.info("Database migration completed successfully!")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Database migration failed: {e}")
        raise
    finally:
        conn.close()

def reset_database():
    """Reset database completely (for development)."""
    import os
    if os.path.exists('trading.db'):
        os.remove('trading.db')
        logger.info("Database reset complete.")
    
    # Reinitialize
    db_manager = DatabaseManager()
    logger.info("New database created with latest schema.")

def check_database_schema():
    """Check current database schema."""
    conn = sqlite3.connect('trading.db')
    cursor = conn.cursor()
    
    try:
        # Check indicators table columns
        cursor.execute("PRAGMA table_info(indicators)")
        columns = cursor.fetchall()
        
        print("📋 Current indicators table schema:")
        print("Column Name | Type")
        print("-" * 30)
        for col in columns:
            print(f"{col[1]:<12} | {col[2]}")
            
        return columns
        
    except Exception as e:
        logger.error(f"Error checking schema: {e}")
        return []
    finally:
        conn.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Database migration tool')
    parser.add_argument('--reset', action='store_true', help='Reset database completely')
    parser.add_argument('--check', action='store_true', help='Check current schema')
    
    args = parser.parse_args()
    
    if args.reset:
        reset_database()
    elif args.check:
        check_database_schema()
    else:
        migrate_database()