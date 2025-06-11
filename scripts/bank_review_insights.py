import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from spacy.cli import download

class BankReviewInsights:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading 'en_core_web_sm' model...")
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        self.theme_palette = {
            'UI/UX': '#4C72B0', 'Reliability': '#DD8452', 'Transactions': '#55A868',
            'Security': '#C44E52', 'Features': '#8172B3', 'Performance': '#937860',
            'Access': '#DA8BC3', 'Customer Service': '#8C8C8C', 'Functionality': '#E9967A',
            'Other': '#999999'
        }

    def preprocess_text(self, text):
        if pd.isna(text):
            return ""
        doc = self.nlp(str(text).lower())
        return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha])

    def analyze_sentiment(self, text):
        analysis = TextBlob(str(text))
        polarity = analysis.sentiment.polarity
        label = "positive" if polarity > 0 else "negative" if polarity < 0 else "neutral"
        return label, polarity

    def extract_keywords(self, texts, ngram_range=(1, 2), max_features=100):
        tfidf = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
        tfidf_matrix = tfidf.fit_transform(texts)
        return tfidf.get_feature_names_out()

    def cluster_themes(self, keywords, bank_name):
        theme_rules = {
            'CBE': {
                'UI/UX': ['ui', 'interface', 'design', 'experience', 'layout'],
                'Reliability': ['crash', 'error', 'bug', 'freeze', 'stable'],
                'Transactions': ['transfer', 'payment', 'send', 'receive', 'money'],
                'Security': ['secure', 'fraud', 'hack', 'protection', 'safe'],
                'Features': ['feature', 'function', 'option', 'tool', 'screenshot']
            },
            'BOA': {
                'Performance': ['slow', 'fast', 'speed', 'lag', 'responsive'],
                'Access': ['login', 'account', 'password', 'access', 'otp'],
                'Customer Service': ['support', 'help', 'service', 'response', 'contact'],
                'Functionality': ['work', 'broken', 'function', 'issue', 'problem']
            },
            'Dashen': {
                'Reliability': ['app', 'crash', 'keeps', 'freezing', 'buggy'],
                'Access': ['login', 'account', 'password', 'access', 'otp'],
                'Functionality': ['work', 'broken', 'function', 'issue', 'problem'],
                'Security': ['secure', 'fraud', 'hack', 'protection', 'safe'],
                'Transactions': ['transfer', 'payment', 'send', 'receive', 'money']
            }
        }

        themes = defaultdict(list)
        for keyword in keywords:
            for theme, terms in theme_rules[bank_name].items():
                if any(term in keyword for term in terms):
                    themes[theme].append(keyword)
                    break
            else:
                themes['Other'].append(keyword)
        return dict(themes)

    def assign_theme(self, text, theme_keywords):
        text = str(text).lower()
        for theme, keywords in theme_keywords.items():
            if any(keyword in text for keyword in keywords):
                return theme
        return "Other"

    def _extract_common_phrases(self, texts):
        vectorizer = TfidfVectorizer(ngram_range=(2, 3), max_features=10)
        vectorizer.fit(texts)
        return vectorizer.get_feature_names_out().tolist()

    def process_and_analyze(self):
        results = []

        for bank_name in self.df['bank'].unique():
            bank_reviews = self.df[self.df['bank'] == bank_name].copy()
            bank_reviews['processed_text'] = bank_reviews['review'].apply(self.preprocess_text)
            bank_reviews[['sentiment_label', 'sentiment_score']] = bank_reviews['review'].apply(
                lambda x: pd.Series(self.analyze_sentiment(x)))
            keywords = self.extract_keywords(bank_reviews['processed_text'])
            theme_keywords = self.cluster_themes(keywords, bank_name)
            bank_reviews['theme'] = bank_reviews['review'].apply(
                lambda x: self.assign_theme(x, theme_keywords))

            for _, row in bank_reviews.iterrows():
                results.append({
                    'review_id': row['reviewId'],
                    'bank': bank_name,
                    'review_text': row['review'],
                    'processed_text': row['processed_text'],
                    'sentiment_label': row['sentiment_label'],
                    'sentiment_score': row['sentiment_score'],
                    'theme': row['theme']
                })

        self.results_df = pd.DataFrame(results)
        self.results_df.to_csv('bank_review_analysis_results.csv', index=False)
        return self.results_df

    def generate_insights(self):
        self.visualize_themes()
        self.visualize_sentiment()

        return {
            'drivers': self.identify_drivers(),
            'pain_points': self.identify_pain_points(),
            'comparisons': self.compare_banks(),
            'improvements': self.suggest_improvements()
        }

    def identify_drivers(self):
        drivers = {}
        for bank in self.results_df['bank'].unique():
            bank_data = self.results_df[self.results_df['bank'] == bank]
            positive_reviews = bank_data[bank_data['sentiment_label'] == 'positive']
            top_themes = positive_reviews['theme'].value_counts().nlargest(2).index.tolist()
            theme_keywords = {theme: self._extract_common_phrases(positive_reviews[positive_reviews['theme'] == theme]['processed_text'])[:3]
                              for theme in top_themes}
            drivers[bank] = theme_keywords
        return drivers

    def identify_pain_points(self):
        pain_points = {}
        for bank in self.results_df['bank'].unique():
            bank_data = self.results_df[self.results_df['bank'] == bank]
            negative_reviews = bank_data[bank_data['sentiment_label'] == 'negative']
            top_themes = negative_reviews['theme'].value_counts().nlargest(2).index.tolist()
            theme_issues = {theme: self._extract_common_phrases(negative_reviews[negative_reviews['theme'] == theme]['processed_text'])[:3]
                            for theme in top_themes}
            pain_points[bank] = theme_issues
        return pain_points

    def compare_banks(self):
        return {
            'theme_distribution': self.results_df.groupby(['bank', 'theme']).size().unstack().fillna(0),
            'sentiment_scores': self.results_df.groupby('bank')['sentiment_score'].mean(),
            'top_strengths': self.identify_drivers(),
            'top_issues': self.identify_pain_points()
        }

    def suggest_improvements(self):
        return {
            'CBE': [
                "Implement screenshot functionality with security safeguards",
                "Add dark mode and UI customization options",
                "Improve transaction history with search/filter capabilities"
            ],
            'BOA': [
                "Enhance app stability and reduce crashes",
                "Simplify account activation and OTP process",
                "Add in-app customer support chat feature"
            ],
            'Dashen': [
                "Enhance app stability and reduce crashes",
                "Simplify account activation and OTP process",
                "Improve transaction history with search/filter capabilities"
            ]
        }

    def visualize_themes(self):
        plt.figure(figsize=(12, 6))
        theme_counts = self.results_df.groupby(['bank', 'theme']).size().unstack()
        (theme_counts.div(theme_counts.sum(axis=1), axis=0) * 100).plot(
            kind='bar',
            stacked=True,
            color=[self.theme_palette.get(t, '#999999') for t in theme_counts.columns],
            width=0.8
        )
        plt.title('Theme Distribution by Bank')
        plt.ylabel('Percentage of Reviews')
        plt.xlabel('Bank')
        plt.xticks(rotation=0)
        plt.legend(title='Themes', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.savefig('theme_distribution.png')
        plt.close()

    def visualize_sentiment(self):
        plt.figure(figsize=(10, 5))
        sns.boxplot(
            data=self.results_df,
            x='bank',
            y='sentiment_score',
            palette=['#4C72B0', '#55A868']
        )
        plt.title('Sentiment Score Distribution by Bank')
        plt.xlabel('Bank')
        plt.ylabel('Sentiment Score (-1 to 1)')
        plt.tight_layout()
        plt.savefig('sentiment_comparison.png')
        plt.close()
