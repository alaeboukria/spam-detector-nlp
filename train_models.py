import os
import joblib
import pandas as pd
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from data_loader import DataLoader
from text_preprocessor import TextPreprocessor


class ModelTrainer:
    def __init__(self):
        self.models_dir = "models"
        self.data_loader = DataLoader()
        self.preprocessor = TextPreprocessor()
        self.models = {}
        self.model_scores = {}

    def ensure_models_dir(self):
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def load_and_prepare_data(self):
        print("\U0001F4CA Chargement des données...")
        df = self.data_loader.load_dataset()
        X = df['message'].values
        y = df['label'].values
        X_vectors = self.preprocessor.fit_vectorizer(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectors, y, test_size=0.2, random_state=42, stratify=y
        )
        print(
            f"✅ Données préparées:\n   - Entraînement: {X_train.shape[0]}\n   - Test: {X_test.shape[0]}\n   - Features: {X_train.shape[1]}")
        return X_train, X_test, y_train, y_test

    def train_naive_bayes(self, X_train, y_train):
        print("\U0001F9E0 Entraînement Naive Bayes...")
        start = time.time()
        model = MultinomialNB(alpha=1.0)
        model.fit(X_train, y_train)
        end = time.time()
        print(f"⏱️ Temps d'entraînement Naive Bayes: {end - start:.2f} secondes")
        return model

    def train_logistic_regression(self, X_train, y_train):
        print("\U0001F9E0 Entraînement Logistic Regression avec plus d’itérations et n-grammes...")
        start = time.time()
        model = LogisticRegression(random_state=42, max_iter=3000, solver='saga', C=1.0)
        model.fit(X_train, y_train)
        end = time.time()
        print(f"⏱️ Temps d'entraînement Logistic Regression: {end - start:.2f} secondes")
        return model

    def train_random_forest(self, X_train, y_train):
        print("\U0001F9E0 Entraînement Random Forest approfondi...")
        start = time.time()
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        end = time.time()
        print(f"⏱️ Temps d'entraînement Random Forest: {end - start:.2f} secondes")
        return model

    def evaluate_model(self, model, X_test, y_test, model_name):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n📈 {model_name}:\n   Précision: {accuracy:.4f}")
        report = classification_report(y_test, y_pred, target_names=['HAM', 'SPAM'])
        print(f"   Rapport détaillé:\n{report}")
        return accuracy, report

    def save_models(self):
        self.ensure_models_dir()
        print("\U0001F4BE Sauvegarde des modèles...")
        for name, model in self.models.items():
            filename = os.path.join(self.models_dir, f"{name}_model.pkl")
            joblib.dump(model, filename)
            print(f"   ✅ {name} sauvegardé")
        vectorizer_path = os.path.join(self.models_dir, "tfidf_vectorizer.pkl")
        joblib.dump(self.preprocessor.vectorizer, vectorizer_path)
        print("   ✅ Vectoriseur sauvegardé")
        scores_path = os.path.join(self.models_dir, "model_scores.pkl")
        joblib.dump(self.model_scores, scores_path)
        print("   ✅ Scores sauvegardés")

    def plot_results(self, X_test, y_test):
        print("\U0001F4CA Génération des graphiques...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Évaluation des Modèles de Détection de Spam', fontsize=16)
        model_names = list(self.model_scores.keys())
        accuracies = list(self.model_scores.values())
        axes[0, 0].bar(model_names, accuracies, color=['skyblue', 'lightgreen', 'salmon'])
        axes[0, 0].set_title('Précision des Modèles')
        axes[0, 0].set_ylabel('Précision')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
        best_model_name = max(self.model_scores, key=self.model_scores.get)
        best_model = self.models[best_model_name]
        y_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['HAM', 'SPAM'],
                    yticklabels=['HAM', 'SPAM'],
                    ax=axes[0, 1])
        axes[0, 1].set_title(f'Matrice de Confusion - {best_model_name}')
        unique, counts = np.unique(y_test, return_counts=True)
        axes[1, 0].pie(counts, labels=['HAM', 'SPAM'], autopct='%1.1f%%',
                       colors=['lightblue', 'lightcoral'])
        axes[1, 0].set_title('Distribution des Classes (Test)')
        axes[1, 1].axis('off')
        comparison_text = "Comparaison des Modèles:\n\n"
        for name, score in sorted(self.model_scores.items(), key=lambda x: x[1], reverse=True):
            comparison_text += f"{name}: {score:.4f}\n"
        axes[1, 1].text(0.1, 0.5, comparison_text, fontsize=12,
                        verticalalignment='center', fontfamily='monospace')
        plt.tight_layout()
        plot_path = os.path.join(self.models_dir, "model_evaluation.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Graphiques sauvegardés: {plot_path}")
        plt.show()

    def train_all_models(self):
        print("\U0001F680 Début de l'entraînement des modèles...")
        print("=" * 50)
        X_train, X_test, y_train, y_test = self.load_and_prepare_data()
        self.models['naive_bayes'] = self.train_naive_bayes(X_train, y_train)
        self.models['logistic_regression'] = self.train_logistic_regression(X_train, y_train)
        self.models['random_forest'] = self.train_random_forest(X_train, y_train)
        print("\n" + "=" * 50)
        print("\U0001F4CA ÉVALUATION DES MODÈLES")
        print("=" * 50)
        for name, model in self.models.items():
            accuracy, _ = self.evaluate_model(model, X_test, y_test, name.replace('_', ' ').title())
            self.model_scores[name] = accuracy
        print("\n" + "=" * 50)
        print("\U0001F3C6 RÉSUMÉ FINAL")
        print("=" * 50)
        best_model = max(self.model_scores, key=self.model_scores.get)
        print(
            f"\U0001F947 Meilleur modèle: {best_model.replace('_', ' ').title()}\n   Précision: {self.model_scores[best_model]:.4f}")
        print(f"\n📊 Classement:")
        for i, (name, score) in enumerate(sorted(self.model_scores.items(),
                                                 key=lambda x: x[1], reverse=True), 1):
            emoji = "\U0001F947" if i == 1 else "\U0001F948" if i == 2 else "\U0001F949"
            print(f"   {emoji} {name.replace('_', ' ').title()}: {score:.4f}")
        self.save_models()
        try:
            self.plot_results(X_test, y_test)
        except ImportError:
            print("⚠️  Matplotlib non disponible, graphiques ignorés")
        print("\n✅ Entraînement terminé avec succès!")
        return self.models, self.model_scores


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_all_models()
