import sqlite3

class Database_connection:
    def __init__(self, none):
        self.df = none

    def create_connection(db_file='bank_reviews.db'):
        """Create a database connection to a SQLite database."""
        conn = sqlite3.connect(db_file)
        return conn

    def create_tables(conn):
        """Create banks and reviews tables."""
        cursor = conn.cursor()

        # Create banks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS banks (
                bank_ID INTEGER PRIMARY KEY,
                reviewId INTEGER NOT NULL,
                bank TEXT NOT NULL,
                country TEXT DEFAULT 'ETH',
                source TEXT
            );
        ''')

        # Create reviews table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reviews (
                review_id INTEGER PRIMARY KEY,
                bank_ID INTEGER NOT NULL,
                user_name TEXT,
                review TEXT,
                rating REAL,
                review_date TEXT,
                sentiment_label TEXT,
                sentiment_score TEXT,
                themes TEXT,
                clean_review TEXT,
                cluster REAL
            );
        ''')

        conn.commit()
        print("Tables created successfully.")

