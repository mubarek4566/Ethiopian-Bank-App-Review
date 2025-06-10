import pandas as pd
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


class ThematicAnalysis:
    def __init__(self, path):
        self.df = path

    def extract_themes(self, column='review'):
        themes = {
            "login_issue": ["login", "sign in", "password", "authentication", "cannot access"],
            "technical_bug": ["crash", "bug", "freeze", "glitch", "slow", "lag"],
            "customer_service": ["customer service", "support", "helpdesk", "agent", "rude"],
            "usability": ["easy", "interface", "navigation", "user-friendly", "confusing"],
            "transactions": ["transfer", "payment", "deposit", "withdraw", "transaction"],
            "features": ["feature", "functionality", "option", "update", "new feature"],
            "performance": ["fast", "slow", "loading", "responsive"]
        }

        def identify_themes(text):
            text = text.lower()
            matched_themes = []

            for theme, keywords in themes.items():
                if any(re.search(rf"\b{kw}\b", text) for kw in keywords):
                    matched_themes.append(theme)
            return matched_themes if matched_themes else ["other"]

        self.df['themes'] = self.df[column].fillna("").apply(identify_themes)

        return self.df

    def preprocess_text(self, text):
        """Preprocess text using NLTK"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Tokenize and lemmatize
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return " ".join(words)


    def extract_keywords(self, text_series, ngram_range=(1, 2), max_features=100):
        """Extract keywords using TF-IDF"""
        tfidf = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
        tfidf_matrix = tfidf.fit_transform(text_series)
        feature_names = tfidf.get_feature_names_out()
        return feature_names
    
    def cluster_keywords(self, keywords, bank_name):
        """Cluster keywords into themes based on predefined rules"""
        themes = {
            'Account Access Issues': ['login', 'error', 'crash', 'bug', 'developer option', 'open', 'access'],
            'Transaction Performance': ['transfer', 'slow', 'fast', 'transaction', 'send money', 'payment', 'deduct'],
            'User Interface & Experience': ['ui', 'interface', 'design', 'easy', 'simple', 'layout', 'experience'],
            'Customer Support': ['support', 'help', 'service', 'contact', 'response', 'assistance'],
            'Feature Requests': ['feature', 'add', 'request', 'screenshot', 'dark mode', 'improvement', 'update']
        }
        # Special rules for each bank based on observed patterns
        if bank_name == 'CBE':
            themes['Security Concerns'] = ['security', 'safe', 'fraud', 'hack', 'protection']
            themes['Network Issues'] = ['network', 'connection', 'sync', 'offline']
        elif bank_name == 'BOA':
            themes['App Stability'] = ['crash', 'bug', 'freeze', 'close', 'unstable']
            themes['Activation Problems'] = ['activate', 'otp', 'initialize', 'registration']
        
        keyword_clusters = defaultdict(list)
        
        for keyword in keywords:
            matched = False
            for theme, terms in themes.items():
                for term in terms:
                    if term in keyword:
                        keyword_clusters[theme].append(keyword)
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                keyword_clusters['Other'].append(keyword)
        
        return dict(keyword_clusters)
    
    def analyze_bank_reviews(self, bank_name):
        """Main function to analyze reviews for a specific bank"""
        print(f"\nAnalyzing reviews for {bank_name}...")
        bank_reviews = self.df[self.df['bank'] == bank_name]['review']
        
        # Preprocess reviews
        processed_reviews = bank_reviews.apply(self.preprocess_text)
        
        # Extract keywords
        keywords = self.extract_keywords(processed_reviews)
        print("\nTop Keywords:")
        print(keywords)
        
        # Cluster keywords
        keyword_clusters = self.cluster_keywords(keywords, bank_name)
        
        print("\nKeyword Clusters:")
        for theme, terms in keyword_clusters.items():
            print(f"\n{theme}:")
            print(", ".join(terms))
        
        return keyword_clusters