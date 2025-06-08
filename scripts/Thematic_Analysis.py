import pandas as pd
import re

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





