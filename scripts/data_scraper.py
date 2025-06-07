# scraper.py

import time
import pandas as pd
from google_play_scraper import Sort, reviews

class ScrapData:
    def __init__(self, path=None):
        self.df = path  

    def fetch_reviews(self, app_id, app_name, n_reviews=400):
        all_reviews = {}
        batch_size = 200
        token = None
        max_attempts = 10
        attempts = 0

        while len(all_reviews) < n_reviews and attempts < max_attempts:
            try:
                result, token = reviews(
                    app_id,
                    lang='en',
                    country='et',
                    sort=Sort.NEWEST,
                    count=batch_size,
                    continuation_token=token
                )

                for r in result:
                    all_reviews[r['reviewId']] = r

                if token is None or not result:
                    break

                attempts += 1
                time.sleep(1)
            except Exception as e:
                print(f"Error fetching {app_name}: {e}")
                break

        print(f"Fetched {len(all_reviews)} unique reviews for {app_name}")
        df = pd.DataFrame(list(all_reviews.values())[:n_reviews])
        df = df[['reviewId', 'userName', 'content', 'score', 'at']]
        df['app_name'] = app_name
        return df
