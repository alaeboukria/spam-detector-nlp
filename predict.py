import os
import joblib
import numpy as np
from text_preprocessor import TextPreprocessor

class SpamPredictor:
    def __init__(self):
        self.models_dir = "models"
        self.models = {}
        self.model_scores = {}
        self.vectorizer = None
        self.preprocessor = TextPreprocessor()
        self.load_models()
    
    def load_models(self):
        """Charger tous les modèles sauvegardés"""
        if not os.path.exists(self.models_dir):
            raise FileNotFoundError(
                f"❌ Dossier des modèles '{self.models_dir}' non trouvé!\n"
                "   Veuillez d'abord entraîner les modèles avec: python main.py --train"
            )
        
        print("📂 Chargement des modèles...")
        
        # Charger les modèles
        model_files = {
            'naive_bayes': 'naive_bayes_model.pkl',
            'logistic_regression': 'logistic_regression_model.pkl',
            'random_forest': 'random_forest_model.pkl'
        }
        
        for name, filename in model_files.items():
            filepath = os.path.join(self.models_dir, filename)
            if os.path.exists(filepath):
                self.models[name] = joblib.load(filepath)
                print(f"   ✅ {name.replace('_', ' ').title()} chargé")
            else:
                print(f"   ⚠️  {filename} non trouvé")
        
        # Charger le vectoriseur
        vectorizer_path = os.path.join(self.models_dir, "tfidf_vectorizer.pkl")
        if os.path.exists(vectorizer_path):
            self.vectorizer = joblib.load(vectorizer_path)
            print("   ✅ Vectoriseur chargé")
        else:
            raise FileNotFoundError("❌ Vectoriseur non trouvé!")
        
        # Charger les scores
        scores_path = os.path.join(self.models_dir, "model_scores.pkl")
        if os.path.exists(scores_path):
            self.model_scores = joblib.load(scores_path)
            print("   ✅ Scores chargés")
        
        if not self.models:
            raise FileNotFoundError("❌ Aucun modèle trouvé!")
        
        print(f"✅ {len(self.models)} modèles chargés avec succès")
    
    def predict_single_email(self, email_text):
        """Prédire si un email est spam ou non"""
        if not email_text.strip():
            return None
        
        # Prétraiter le texte
        processed_text = self.preprocessor.preprocess_text(email_text)
        
        # Vectoriser
        text_vector = self.vectorizer.transform([processed_text])
        
        # Prédictions de chaque modèle
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            # Prédiction
            pred = model.predict(text_vector)[0]
            predictions[name] = pred
            
            # Probabilités
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(text_vector)[0]
                probabilities[name] = {
                    'ham': proba[0],
                    'spam': proba[1],
                    'confidence': max(proba)
                }
            else:
                # Pour les modèles sans predict_proba
                probabilities[name] = {
                    'ham': 1 - pred,
                    'spam': pred,
                    'confidence': 0.8  # Valeur par défaut
                }
        
        return predictions, probabilities
    
    def get_consensus_prediction(self, predictions, probabilities):
        """Obtenir la prédiction consensuelle"""
        # Compter les votes
        spam_votes = sum(1 for pred in predictions.values() if pred == 1)
        total_votes = len(predictions)
        
        # Décision majoritaire
        is_spam = spam_votes > total_votes / 2
        
        # Calculer la confiance pondérée
        weighted_confidence = 0
        total_weight = 0
        
        for name, pred in predictions.items():
            model_accuracy = self.model_scores.get(name, 0.8)  # Défaut si pas de score
            confidence = probabilities[name]['confidence']
            
            # Pondérer par la précision du modèle
            weight = model_accuracy
            weighted_confidence += confidence * weight
            total_weight += weight
        
        final_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.5
        
        # Type de consensus
        if spam_votes == total_votes:
            consensus_type = "UNANIMITÉ"
        elif spam_votes == 0:
            consensus_type = "UNANIMITÉ"
        elif spam_votes > total_votes / 2:
            consensus_type = "MAJORITÉ"
        else:
            consensus_type = "DIVISION"
        
        return {
            'is_spam': is_spam,
            'confidence': final_confidence,
            'spam_votes': spam_votes,
            'total_votes': total_votes,
            'consensus_type': consensus_type
        }
    
    def analyze_email(self, email_text):
        """Analyser complètement un email"""
        if not email_text.strip():
            return {
                'error': "Texte vide",
                'message': "Veuillez fournir un texte d'email à analyser."
            }
        
        try:
            # Prédictions individuelles
            predictions, probabilities = self.predict_single_email(email_text)
            
            # Consensus
            consensus = self.get_consensus_prediction(predictions, probabilities)
            
            # Analyse des caractéristiques du texte
            text_features = self.preprocessor.analyze_text_features(email_text)
            
            return {
                'email_text': email_text,
                'predictions': predictions,
                'probabilities': probabilities,
                'consensus': consensus,
                'text_features': text_features,
                'model_scores': self.model_scores
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'message': "Erreur lors de l'analyse de l'email."
            }
    
    def format_analysis_result(self, result):
        """Formater le résultat d'analyse pour l'affichage"""
        if 'error' in result:
            return f"❌ ERREUR: {result['message']}"
        
        consensus = result['consensus']
        predictions = result['predictions']
        probabilities = result['probabilities']
        model_scores = result['model_scores']
        
        # En-tête du résultat
        spam_emoji = "🚨" if consensus['is_spam'] else "✅"
        result_text = "SPAM" if consensus['is_spam'] else "HAM"
        
        output = []
        output.append("=" * 60)
        output.append(f"{spam_emoji} RÉSULTAT: {result_text} (Confiance: {consensus['confidence']:.1%})")
        output.append(f"📊 CONSENSUS: {consensus['consensus_type']} ({consensus['spam_votes']}/{consensus['total_votes']} modèles)")
        output.append("=" * 60)
        
        # Détails par modèle
        output.append("\nDÉTAILS PAR MODÈLE:")
        output.append("-" * 40)
        
        for name, pred in predictions.items():
            model_name = name.replace('_', ' ').title()
            pred_emoji = "🚨" if pred == 1 else "✅"
            pred_text = "SPAM" if pred == 1 else "HAM"
            
            proba = probabilities[name]
            confidence = proba['confidence']
            spam_proba = proba['spam']
            ham_proba = proba['ham']
            
            model_accuracy = model_scores.get(name, 0.0)
            
            output.append(f"\n{pred_emoji} {model_name}:")
            output.append(f"   Prédiction: {pred_text} (Confiance: {confidence:.1%})")
            output.append(f"   Probabilité SPAM: {spam_proba:.1%}")
            output.append(f"   Probabilité HAM: {ham_proba:.1%}")
            output.append(f"   Précision du modèle: {model_accuracy:.1%}")
        
        # Caractéristiques du texte
        features = result['text_features']
        output.append(f"\n📝 CARACTÉRISTIQUES DU TEXTE:")
        output.append("-" * 40)
        output.append(f"   Longueur: {features['length']} caractères")
        output.append(f"   Nombre de mots: {features['word_count']}")
        output.append(f"   Points d'exclamation: {features['exclamation_count']}")
        output.append(f"   Points d'interrogation: {features['question_count']}")
        output.append(f"   Ratio majuscules: {features['uppercase_ratio']:.1%}")
        output.append(f"   Chiffres: {features['digit_count']}")
        output.append(f"   Caractères spéciaux: {features['special_char_count']}")
        
        output.append("=" * 60)
        
        return "\n".join(output)
    
    def get_model_status(self):
        """Obtenir le statut des modèles"""
        status = {
            'models_loaded': len(self.models),
            'models_available': list(self.models.keys()),
            'vectorizer_loaded': self.vectorizer is not None,
            'scores_available': bool(self.model_scores)
        }
        
        if self.model_scores:
            best_model = max(self.model_scores, key=self.model_scores.get)
            status['best_model'] = best_model
            status['best_accuracy'] = self.model_scores[best_model]
        
        return status

if __name__ == "__main__":
    # Test rapide
    predictor = SpamPredictor()
    
    test_emails = [
        "Congratulations! You won $1000! Click here now!",
        "Hi, how are you doing today? Let's meet for coffee.",
        "URGENT: Your account will be closed. Verify now!"
    ]
    
    for email in test_emails:
        print(f"\n📧 Test: {email[:50]}...")
        result = predictor.analyze_email(email)
        print(predictor.format_analysis_result(result))
