import sqlite3
import pandas as pd

class insert_data:
    def __init__(self, path):
        self.df = path

    def insert_reviews_from_csv(self, db_file='bank_reviews.db'):
        # Step 1: Connect to the SQLite database
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # Step 2: Read the CSV file
        df = pd.read_csv(self.df)

        # Step 3: Insert rows into the reviews table
        for _, row in df.iterrows():
            cursor.execute('''
                INSERT INTO reviews (
                    reviewId, review, rating, clean_review, review_date, sentiment_label, 
                    sentiment_score, themes, cluster, bank_ID
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                int(row['reviewId']),
                row['review'],
                row['rating'],
                row['clean_review'],
                row['review_date'],
                row['sentiment_label'],
                row['sentiment_score'],
                row['themes'],
                float(row['cluster']),
                1  # <-- default bank_ID for now; change if needed
            ))

        conn.commit()
        conn.close()
        print("CSV data inserted into reviews table successfully.")

       ['reviewId', 'review', 'clean_review', 'sentiment_label',
       'sentiment_score', 'themes', 'cluster'] 
