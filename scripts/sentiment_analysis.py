import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import pipeline
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

class sentimentAnalysis:
    def __init__(self, path):
        self.df = path
    
    def visualize_results(agg_results):
        """Create visualizations of sentiment analysis results"""
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Sentiment distribution by bank and rating
        plt.figure(figsize=(12, 8))
        sns.barplot(data=agg_results, x='rating', y='mean_sentiment', hue='bank')
        plt.title('Average Sentiment by Bank and Star Rating')
        plt.ylabel('Average Sentiment Score (0-1)')
        plt.xlabel('Star Rating')
        plt.legend(title='Bank')
        plt.tight_layout()
        plt.savefig('figures/sentiment_by_bank_rating.png')
        plt.close()
        
        # 2. Review count by rating and bank
        plt.figure(figsize=(12, 8))
        sns.barplot(data=agg_results, x='rating', y='count', hue='bank')
        plt.title('Review Count by Bank and Star Rating')
        plt.ylabel('Number of Reviews')
        plt.xlabel('Star Rating')
        plt.legend(title='Bank')
        plt.tight_layout()
        plt.savefig('figures/review_count_by_bank_rating.png')
        plt.close()

        # 3. Heatmap of sentiment by bank and rating
        pivot_table = agg_results.pivot(index='bank', columns='rating', values='mean_sentiment')
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, cmap='coolwarm', center=0.5, vmin=0, vmax=1)
        plt.title('Sentiment Heatmap by Bank and Rating')
        plt.ylabel('Bank')
        plt.xlabel('Star Rating')
        plt.tight_layout()
        plt.savefig('figures/sentiment_heatmap.png')
        plt.close()
    


    def analyze_sentiment_vader(self, column='review'):
        sia = SentimentIntensityAnalyzer()
        sentiments = []

        for text in self.df[column]:
            if not isinstance(text, str) or text.strip() == '':
                sentiments.append("neutral")
                continue

            score = sia.polarity_scores(text)
            compound = score['compound']
            if compound >= 0.05:
                sentiments.append('positive')
            elif compound <= -0.05:
                sentiments.append('negative')
            else:
                sentiments.append('neutral')

        self.df['sentiment_vader'] = sentiments
        return self.df

    def analyze_sentiment_bert(self, column='review', batch_size=32):
        sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

        sentiments = []
        for i in range(0, len(self.df), batch_size):
            batch = self.df[column].iloc[i:i+batch_size].fillna("").tolist()
            results = sentiment_pipeline(batch)
            for result in results:
                label = result['label'].lower()  # returns 'POSITIVE' or 'NEGATIVE'
                sentiments.append(label)

        self.df['sentiment_bert'] = sentiments
        return self.df
    
    from transformers import pipeline

    def analyze_sentiment_bert1(self, column='review', batch_size=32):
        sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", framework="pt")

        sentiments = []
        for i in range(0, len(self.df), batch_size):
            batch = self.df[column].iloc[i:i+batch_size].fillna("").tolist()
            results = sentiment_pipeline(batch)
            for result in results:
                label = result['label'].lower()  # returns 'positive' or 'negative'
                sentiments.append(label)

        self.df['sentiment_bert'] = sentiments
        return self.df

    def sentiment_by_bank_and_rating(self):
        # Convert sentiment labels to numeric scores
        self.df['sentiment_score'] = self.df['sentiment_bert'].map({'positive': 1, 'neutral': 0, 'negative': -1})
        
        # Group by bank and rating, then calculate mean sentiment score
        agg_df = self.df.groupby(['bank', 'rating'])['sentiment_score'].mean().reset_index()

        # Rename the column for clarity
        agg_df.rename(columns={'sentiment_score': 'mean_sentiment'}, inplace=True)

        return agg_df
    
    def compare_star_rating_with_bert_sentiment(self, rating_column='rating'):
        # Map numeric ratings to sentiment categories
        def map_rating_to_sentiment(rating):
            if rating in [4, 5]:
                return 'positive'
            elif rating == 3:
                return 'neutral'
            else:
                return 'negative'

        self.df['sentiment_rating_based'] = self.df[rating_column].apply(map_rating_to_sentiment)

        # Filter only rows where model-predicted sentiment exists
        comparison_df = self.df.dropna(subset=['sentiment_bert', 'sentiment_rating_based'])

        # Compare and generate confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report

        # Remove neutral from comparison since BERT only gives positive/negative
        filtered = comparison_df[comparison_df['sentiment_rating_based'] != 'neutral']
        y_true = filtered['sentiment_rating_based']
        y_pred = filtered['sentiment_bert']

        print("Classification Report (excluding neutral ratings):\n")
        print(classification_report(y_true, y_pred))

        # Return DataFrame with sentiment comparison for further analysis
        return filtered[['bank', 'review', 'rating', 'sentiment_rating_based', 'sentiment_bert']]

