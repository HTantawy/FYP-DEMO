# database.py
import psycopg2
import streamlit as st
from auth_app import DB_CONFIG, init_db

def verify_database():
    """Verify database connection and table existence"""
    conn = None  # Initialize conn to None before the try block
    cur = None   # Initialize cur to None as well
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Check if tables exist
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'users'
            );
        """)
        users_exists = cur.fetchone()[0]
        
        if not users_exists:
            print("Users table not found, initializing database...")
            init_db()
        else:
            print("Database verification successful!")
            
    except Exception as e:
        print(f"Database verification failed: {str(e)}")
        init_db()
        
    finally:
        if cur:  # Check if cur exists before closing it
            cur.close()
        if conn:  # Now this check is safe because conn is always defined
            conn.close()