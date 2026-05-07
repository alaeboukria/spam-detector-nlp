

#!/usr/bin/env python3
"""
🚨 DÉTECTEUR DE SPAM - Machine Learning
=====================================

Application complète de détection de spams utilisant 3 modèles ML:
- Naive Bayes
- Logistic Regression  
- Random Forest

Auteur: Assistant IA
Version: 1.0
"""

import sys
import os
import argparse
from train_models import ModelTrainer
from predict import SpamPredictor
from spam_detector_gui import SpamDetectorGUI

def print_banner():
    """Afficher la bannière de l'application"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    🚨 DÉTECTEUR DE SPAM 🚨                    ║
║                                                              ║
║              Intelligence Artificielle Avancée              ║
║                   3 Modèles Machine Learning                 ║
║                                                              ║
║  • Naive Bayes        • Logistic Regression  • Random Forest ║
║  • Analyse NLP        • Vote Majoritaire     • Interface GUI ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_help():
    """Afficher l'aide détaillée"""
    help_text = """
🎯 UTILISATION:

1. ENTRAÎNER LES MODÈLES (obligatoire en premier):
   python main.py --train

2. MODE INTERACTIF (recommandé):
   python main.py --predict

3. INTERFACE GRAPHIQUE:
   python main.py --gui

4. ANALYSER UN TEXTE DIRECTEMENT:
   python main.py --text "Votre texte ici"

5. ANALYSER UN FICHIER:
   python main.py --file chemin/vers/email.txt

6. TESTS AUTOMATIQUES:
   python main.py --test

7. STATUT DU SYSTÈME:
   python main.py --status

📁 STRUCTURE DU PROJET:
   models/          - Modèles entraînés (.pkl)
   data/           - Dataset de spams
   *.py            - Scripts Python

🔧 DÉPENDANCES:
   pip install -r requirements.txt

💡 CONSEILS:
   - Entraînez d'abord les modèles avec --train
   - Utilisez --gui pour une interface conviviale
   - Le mode --predict permet l'analyse interactive
   - Les résultats incluent la confiance de chaque modèle
    """
    print(help_text)

def train_models():
    """Entraîner tous les modèles"""
    print("🚀 ENTRAÎNEMENT DES MODÈLES")
    print("=" * 50)
    
    try:
        trainer = ModelTrainer()
        models, scores = trainer.train_all_models()
        
        print("\n🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        print("\nVous pouvez maintenant utiliser:")
        print("  python main.py --predict    (mode interactif)")
        print("  python main.py --gui        (interface graphique)")
        print("  python main.py --test       (tests automatiques)")
        
        return True
        
    except Exception as e:
        print(f"❌ ERREUR lors de l'entraînement: {e}")
        return False

def interactive_predict():
    """Mode prédiction interactif"""
    print("🔍 MODE PRÉDICTION INTERACTIF")
    print("=" * 50)
    print("Tapez 'quit' pour quitter, 'help' pour l'aide")
    print()
    
    try:
        predictor = SpamPredictor()
        
        while True:
            print("\n" + "─" * 50)
            email_text = input("📧 Entrez l'email à analyser: ").strip()
            
            if email_text.lower() in ['quit', 'exit', 'q']:
                print("👋 Au revoir!")
                break
            elif email_text.lower() == 'help':
                print("\n💡 AIDE:")
                print("  - Collez le contenu d'un email")
                print("  - Tapez 'quit' pour quitter")
                print("  - Tapez 'test' pour des exemples")
                continue
            elif email_text.lower() == 'test':
                run_tests()
                continue
            elif not email_text:
                print("⚠️  Veuillez entrer un texte d'email.")
                continue
            
            # Analyser l'email
            print("\n🔄 Analyse en cours...")
            result = predictor.analyze_email(email_text)
            formatted_result = predictor.format_analysis_result(result)
            print(formatted_result)
            
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\n💡 Solution: Entraînez d'abord les modèles avec:")
        print("   python main.py --train")
    except KeyboardInterrupt:
        print("\n\n👋 Arrêt demandé par l'utilisateur. Au revoir!")
    except Exception as e:
        print(f"❌ ERREUR: {e}")

def analyze_text(text):
    """Analyser un texte donné"""
    try:
        predictor = SpamPredictor()
        result = predictor.analyze_email(text)
        formatted_result = predictor.format_analysis_result(result)
        print(formatted_result)
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\n💡 Solution: Entraînez d'abord les modèles avec:")
        print("   python main.py --train")
    except Exception as e:
        print(f"❌ ERREUR: {e}")

def analyze_file(filepath):
    """Analyser un fichier"""
    try:
        if not os.path.exists(filepath):
            print(f"❌ Fichier non trouvé: {filepath}")
            return
        
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        
        print(f"📁 Analyse du fichier: {filepath}")
        print("=" * 50)
        
        analyze_text(content)
        
    except Exception as e:
        print(f"❌ ERREUR lors de la lecture du fichier: {e}")

def run_tests():
    """Exécuter des tests automatiques"""
    print("🧪 TESTS AUTOMATIQUES")
    print("=" * 50)
    
    test_emails = [
        {
            'text': "Congratulations! You won $1,000,000! Click here immediately to claim your prize! Send your bank details to winner@lottery.com",
            'expected': 'SPAM',
            'description': 'Email de loterie frauduleuse'
        },
        {
            'text': "URGENT: Your account will be suspended! Verify your identity now by clicking this link and entering your password!",
            'expected': 'SPAM', 
            'description': 'Tentative de phishing'
        },
        {
            'text': "FREE MONEY! CALL NOW! Limited time offer! Don't miss out! Act fast!",
            'expected': 'SPAM',
            'description': 'Offre d\'argent gratuit'
        },
        {
            'text': "Hi John, how are you doing? Let's meet for coffee tomorrow at 3 PM. Looking forward to catching up!",
            'expected': 'HAM',
            'description': 'Message personnel légitime'
        },
        {
            'text': "Your meeting with the client has been rescheduled to Friday at 2 PM. Please confirm your attendance.",
            'expected': 'HAM',
            'description': 'Email professionnel'
        },
        {
            'text': "Thank you for your purchase. Your order #12345 has been shipped and will arrive in 3-5 business days.",
            'expected': 'HAM',
            'description': 'Confirmation de commande'
        }
    ]
    
    try:
        predictor = SpamPredictor()
        correct_predictions = 0
        
        for i, test in enumerate(test_emails, 1):
            print(f"\n📧 TEST {i}: {test['description']}")
            print(f"Texte: {test['text'][:60]}...")
            print(f"Attendu: {test['expected']}")
            
            result = predictor.analyze_email(test['text'])
            
            if 'error' not in result:
                predicted = "SPAM" if result['consensus']['is_spam'] else "HAM"
                confidence = result['consensus']['confidence']
                
                print(f"Prédit: {predicted} ({confidence:.1%} confiance)")
                
                if predicted == test['expected']:
                    print("✅ CORRECT")
                    correct_predictions += 1
                else:
                    print("❌ INCORRECT")
            else:
                print(f"❌ ERREUR: {result['message']}")
        
        print(f"\n📊 RÉSULTATS DES TESTS:")
        print(f"   Réussis: {correct_predictions}/{len(test_emails)}")
        print(f"   Précision: {correct_predictions/len(test_emails):.1%}")
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\n💡 Solution: Entraînez d'abord les modèles avec:")
        print("   python main.py --train")
    except Exception as e:
        print(f"❌ ERREUR: {e}")

def show_status():
    """Afficher le statut du système"""
    print("📊 STATUT DU SYSTÈME")
    print("=" * 50)
    
    try:
        # Vérifier les modèles
        models_dir = "models"
        if os.path.exists(models_dir):
            print("✅ Dossier des modèles trouvé")
            
            model_files = [
                'naive_bayes_model.pkl',
                'logistic_regression_model.pkl', 
                'random_forest_model.pkl',
                'tfidf_vectorizer.pkl',
                'model_scores.pkl'
            ]
            
            for file in model_files:
                filepath = os.path.join(models_dir, file)
                if os.path.exists(filepath):
                    size = os.path.getsize(filepath)
                    print(f"   ✅ {file} ({size/1024:.1f} KB)")
                else:
                    print(f"   ❌ {file} (manquant)")
        else:
            print("❌ Dossier des modèles non trouvé")
        
        # Vérifier les données
        data_dir = "data"
        if os.path.exists(data_dir):
            print("\n✅ Dossier des données trouvé")
            dataset_path = os.path.join(data_dir, "spam_dataset.csv")
            if os.path.exists(dataset_path):
                size = os.path.getsize(dataset_path)
                print(f"   ✅ spam_dataset.csv ({size/1024:.1f} KB)")
            else:
                print("   ❌ spam_dataset.csv (manquant)")
        else:
            print("\n❌ Dossier des données non trouvé")
        
        # Tester le chargement des modèles
        print("\n🔧 Test de chargement des modèles...")
        try:
            predictor = SpamPredictor()
            status = predictor.get_model_status()
            
            print(f"   ✅ {status['models_loaded']} modèles chargés")
            print(f"   ✅ Vectoriseur: {'OK' if status['vectorizer_loaded'] else 'ERREUR'}")
            
            if 'best_model' in status:
                best_model = status['best_model'].replace('_', ' ').title()
                accuracy = status['best_accuracy']
                print(f"   🏆 Meilleur modèle: {best_model} ({accuracy:.1%})")
            
            print("\n🎉 SYSTÈME OPÉRATIONNEL!")
            print("\nCommandes disponibles:")
            print("  python main.py --predict    (mode interactif)")
            print("  python main.py --gui        (interface graphique)")
            print("  python main.py --test       (tests automatiques)")
            
        except Exception as e:
            print(f"   ❌ Erreur de chargement: {e}")
            print("\n💡 Solution: Entraînez les modèles avec:")
            print("   python main.py --train")
            
    except Exception as e:
        print(f"❌ ERREUR: {e}")

def launch_gui():
    """Lancer l'interface graphique"""
    try:
        print("🖥️  Lancement de l'interface graphique...")
        app = SpamDetectorGUI()
        app.run()
    except ImportError:
        print("❌ ERREUR: tkinter non disponible")
        print("Sur Ubuntu/Debian: sudo apt-get install python3-tk")
        print("Sur macOS: brew install python-tk")
    except Exception as e:
        print(f"❌ ERREUR GUI: {e}")

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(
        description="🚨 Détecteur de Spam - Machine Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python main.py --train                    # Entraîner les modèles
  python main.py --predict                  # Mode interactif
  python main.py --gui                      # Interface graphique
  python main.py --text "Spam text here"   # Analyser un texte
  python main.py --file email.txt          # Analyser un fichier
  python main.py --test                     # Tests automatiques
  python main.py --status                   # Statut du système
        """
    )
    
    parser.add_argument('--train', action='store_true', 
                       help='Entraîner les modèles ML')
    parser.add_argument('--predict', action='store_true',
                       help='Mode prédiction interactif')
    parser.add_argument('--gui', action='store_true',
                       help='Lancer l\'interface graphique')
    parser.add_argument('--text', type=str,
                       help='Analyser un texte directement')
    parser.add_argument('--file', type=str,
                       help='Analyser un fichier texte')
    parser.add_argument('--test', action='store_true',
                       help='Exécuter des tests automatiques')
    parser.add_argument('--status', action='store_true',
                       help='Afficher le statut du système')
    
    args = parser.parse_args()
    
    # Si aucun argument, afficher l'aide
    if len(sys.argv) == 1:
        print_banner()
        print_help()
        return
    
    print_banner()
    
    # Exécuter l'action demandée
    if args.train:
        train_models()
    elif args.predict:
        interactive_predict()
    elif args.gui:
        launch_gui()
    elif args.text:
        analyze_text(args.text)
    elif args.file:
        analyze_file(args.file)
    elif args.test:
        run_tests()
    elif args.status:
        show_status()
    else:
        print("❌ Action non reconnue. Utilisez --help pour voir les options.")

if __name__ == "__main__":
    main()
