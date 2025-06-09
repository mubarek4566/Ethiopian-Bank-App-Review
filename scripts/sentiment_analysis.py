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
        self.agg_results = None
    
    def analyze_sentiment1(self, model_name="distilbert-base-uncased-finetuned-sst-2-english", batch_size=32):
        """Perform sentiment analysis using HuggingFace pipeline"""
        # Initialize sentiment analysis pipeline (simplified version)
        sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model=model_name,
            device=-1  # Force CPU usage
        )
        
        # Process reviews in batches
        reviews = self.df['review'].tolist()
        sentiments = []
        
        print("Performing sentiment analysis...")
        for i in tqdm(range(0, len(reviews), batch_size)):
            batch = reviews[i:i+batch_size]
            try:
                results = sentiment_pipeline(batch)
                sentiments.extend(results)
            except Exception as e:
                print(f"Error processing batch {i}: {str(e)}")
                # Fill with neutral sentiment if error occurs
                sentiments.extend([{'label': 'NEUTRAL', 'score': 0.5}] * len(batch))
        
        # Extract sentiment scores
        self.df['sentiment_label'] = [s['label'] for s in sentiments]
        self.df['sentiment_score'] = [s['score'] for s in sentiments]
        
        # Convert to numeric sentiment (1 for POSITIVE, 0 for NEGATIVE, 0.5 for NEUTRAL)
        self.df['sentiment_numeric'] = self.df['sentiment_label'].apply(
            lambda x: 1 if x == 'POSITIVE' else (0 if x == 'NEGATIVE' else 0.5)
        )
            
        return self.df
    
    def aggregate_sentiment1(self):
        """Aggregate sentiment scores by bank and rating"""
        # Group by bank and rating
        agg_results = self.df.groupby(['bank', 'rating']).agg(
            mean_sentiment=('sentiment_numeric', 'mean'),
            count=('reviewId', 'count'),
            mean_sentiment_score=('sentiment_score', 'mean')
        ).reset_index()
        
        # Add sentiment category based on mean
        agg_results['sentiment_category'] = pd.cut(
            agg_results['mean_sentiment'],
            bins=[-0.1, 0.33, 0.66, 1.1],
            labels=['Negative', 'Neutral', 'Positive']
        )
        
        return agg_results


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

    def visualize_vader_sentiment(self):
        # Set style
        sns.set(style="whitegrid")

        # 1. Countplot of overall sentiment distribution
        plt.figure(figsize=(8, 5))
        sns.countplot(data=self.df, x='sentiment_vader', order=['positive', 'neutral', 'negative'], palette='Set2')
        plt.title("Overall VADER Sentiment Distribution")
        plt.xlabel("Sentiment")
        plt.ylabel("Number of Reviews")
        plt.tight_layout()
        plt.show()

        # 2. Sentiment distribution by bank
        plt.figure(figsize=(12, 6))
        sns.countplot(data=self.df, x='bank', hue='sentiment_vader', order=self.df['bank'].value_counts().index, palette='Set1')
        plt.title("VADER Sentiment Distribution by Bank")
        plt.xlabel("Bank")
        plt.ylabel("Number of Reviews")
        plt.xticks(rotation=45)
        plt.legend(title="Sentiment")
        plt.tight_layout()
        plt.show()

        # 3. Pie chart (optional)
        sentiment_counts = self.df['sentiment_vader'].value_counts()
        plt.figure(figsize=(6, 6))
        sentiment_counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2"))
        plt.title("VADER Sentiment Proportions")
        plt.ylabel("")  # Hide y-label
        plt.tight_layout()
        plt.show()


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

