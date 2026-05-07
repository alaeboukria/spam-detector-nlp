import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

class TextPreprocessor:
    def __init__(self):
        self.download_nltk_data()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
        
    def download_nltk_data(self):
        """Télécharger les ressources NLTK nécessaires"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("📥 Téléchargement des ressources NLTK...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            print("✅ Ressources NLTK téléchargées")
    
    def clean_text(self, text):
        """Nettoyer le texte"""
        if not isinstance(text, str):
            return ""
        
        # Convertir en minuscules
        text = text.lower()
        
        # Supprimer les URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Supprimer les adresses email
        text = re.sub(r'\S+@\S+', '', text)
        
        # Supprimer les numéros de téléphone
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        # Supprimer les balises HTML
        text = re.sub(r'<.*?>', '', text)
        
        # Supprimer les caractères spéciaux et chiffres
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Supprimer les espaces multiples
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize_and_lemmatize(self, text):
        """Tokeniser et lemmatiser le texte"""
        # Tokenisation
        tokens = word_tokenize(text)
        
        # Supprimer les stopwords et lemmatiser
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(tokens)
    
    def preprocess_text(self, text):
        """Pipeline complet de prétraitement"""
        # Nettoyage
        cleaned = self.clean_text(text)
        
        # Tokenisation et lemmatisation
        processed = self.tokenize_and_lemmatize(cleaned)
        
        return processed
    
    def fit_vectorizer(self, texts):
        """Entraîner le vectoriseur TF-IDF"""
        print("🔧 Entraînement du vectoriseur TF-IDF...")
        
        # Prétraiter tous les textes
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Créer et entraîner le vectoriseur
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # Unigrammes et bigrammes
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        
        vectors = self.vectorizer.fit_transform(processed_texts)
        print(f"✅ Vectoriseur entraîné: {vectors.shape[1]} features")
        
        return vectors
    
    def transform_text(self, texts):
        """Transformer les textes en vecteurs"""
        if self.vectorizer is None:
            raise ValueError("Le vectoriseur n'est pas entraîné. Utilisez fit_vectorizer() d'abord.")
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Prétraiter les textes
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Vectoriser
        return self.vectorizer.transform(processed_texts)
    
    def get_feature_names(self):
        """Obtenir les noms des features"""
        if self.vectorizer:
            return self.vectorizer.get_feature_names_out()
        return []
    
    def analyze_text_features(self, text):
        """Analyser les caractéristiques d'un texte"""
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'digit_count': sum(1 for c in text if c.isdigit()),
            'special_char_count': sum(1 for c in text if c in string.punctuation)
        }
        return features
