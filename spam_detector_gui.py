import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
from predict import SpamPredictor

class SpamDetectorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🚨 Détecteur de Spam - Machine Learning")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        # Initialiser le prédicteur
        self.predictor = None
        self.load_predictor()
        
        self.create_widgets()
        
    def load_predictor(self):
        """Charger le prédicteur en arrière-plan"""
        try:
            self.predictor = SpamPredictor()
        except Exception as e:
            messagebox.showerror(
                "Erreur", 
                f"Impossible de charger les modèles:\n{str(e)}\n\n"
                "Veuillez d'abord entraîner les modèles avec:\n"
                "python main.py --train"
            )
            self.root.quit()
    
    def create_widgets(self):
        """Créer l'interface utilisateur"""
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Titre principal
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill='x', padx=0, pady=0)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="🚨 DÉTECTEUR DE SPAM", 
            font=('Arial', 20, 'bold'),
            fg='white', 
            bg='#2c3e50'
        )
        title_label.pack(expand=True)
        
        subtitle_label = tk.Label(
            title_frame, 
            text="Intelligence Artificielle • 3 Modèles ML • Analyse Avancée", 
            font=('Arial', 10),
            fg='#ecf0f1', 
            bg='#2c3e50'
        )
        subtitle_label.pack()
        
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Section d'entrée
        input_frame = tk.LabelFrame(
            main_frame, 
            text="📧 Email à analyser", 
            font=('Arial', 12, 'bold'),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        input_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        # Zone de texte
        self.text_area = scrolledtext.ScrolledText(
            input_frame,
            height=8,
            font=('Consolas', 11),
            wrap=tk.WORD,
            bg='white',
            fg='#2c3e50',
            insertbackground='#3498db'
        )
        self.text_area.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Placeholder
        placeholder_text = (
            "Collez ici le contenu de l'email à analyser...\n\n"
            "Exemple:\n"
            "Congratulations! You have won $1,000,000!\n"
            "Click here immediately to claim your prize!\n"
            "Send your bank details to winner@lottery.com"
        )
        self.text_area.insert('1.0', placeholder_text)
        self.text_area.config(fg='gray')
        
        # Événements pour le placeholder
        self.text_area.bind('<FocusIn>', self.on_text_focus_in)
        self.text_area.bind('<FocusOut>', self.on_text_focus_out)
        
        # Boutons
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(fill='x', pady=(0, 10))
        
        # Bouton analyser
        self.analyze_button = tk.Button(
            button_frame,
            text="🔍 ANALYSER L'EMAIL",
            font=('Arial', 12, 'bold'),
            bg='#3498db',
            fg='white',
            activebackground='#2980b9',
            activeforeground='white',
            relief='flat',
            padx=20,
            pady=10,
            command=self.analyze_email_threaded
        )
        self.analyze_button.pack(side='left', padx=(0, 10))
        
        # Bouton charger fichier
        load_button = tk.Button(
            button_frame,
            text="📁 Charger fichier",
            font=('Arial', 10),
            bg='#95a5a6',
            fg='white',
            activebackground='#7f8c8d',
            activeforeground='white',
            relief='flat',
            padx=15,
            pady=10,
            command=self.load_file
        )
        load_button.pack(side='left', padx=(0, 10))
        
        # Bouton effacer
        clear_button = tk.Button(
            button_frame,
            text="🗑️ Effacer",
            font=('Arial', 10),
            bg='#e74c3c',
            fg='white',
            activebackground='#c0392b',
            activeforeground='white',
            relief='flat',
            padx=15,
            pady=10,
            command=self.clear_text
        )
        clear_button.pack(side='left')
        
        # Barre de progression
        self.progress = ttk.Progressbar(
            button_frame, 
            mode='indeterminate',
            length=200
        )
        self.progress.pack(side='right')
        
        # Section de résultats
        result_frame = tk.LabelFrame(
            main_frame, 
            text="📊 Résultats de l'analyse", 
            font=('Arial', 12, 'bold'),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        result_frame.pack(fill='both', expand=True)
        
        # Zone de résultats
        self.result_area = scrolledtext.ScrolledText(
            result_frame,
            height=12,
            font=('Consolas', 10),
            wrap=tk.WORD,
            bg='#2c3e50',
            fg='#ecf0f1',
            insertbackground='#3498db',
            state='disabled'
        )
        self.result_area.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Barre de statut
        status_frame = tk.Frame(self.root, bg='#34495e', height=30)
        status_frame.pack(fill='x', side='bottom')
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(
            status_frame,
            text="✅ Prêt - Modèles chargés",
            font=('Arial', 9),
            fg='#ecf0f1',
            bg='#34495e'
        )
        self.status_label.pack(side='left', padx=10, pady=5)
        
        # Afficher le statut des modèles
        if self.predictor:
            status = self.predictor.get_model_status()
            status_text = f"✅ {status['models_loaded']} modèles chargés"
            if 'best_model' in status:
                best_model = status['best_model'].replace('_', ' ').title()
                accuracy = status['best_accuracy']
                status_text += f" • Meilleur: {best_model} ({accuracy:.1%})"
            self.status_label.config(text=status_text)
    
    def on_text_focus_in(self, event):
        """Gérer le focus sur la zone de texte"""
        if self.text_area.get('1.0', 'end-1c').startswith('Collez ici'):
            self.text_area.delete('1.0', 'end')
            self.text_area.config(fg='#2c3e50')
    
    def on_text_focus_out(self, event):
        """Gérer la perte de focus sur la zone de texte"""
        if not self.text_area.get('1.0', 'end-1c').strip():
            placeholder_text = (
                "Collez ici le contenu de l'email à analyser...\n\n"
                "Exemple:\n"
                "Congratulations! You have won $1,000,000!\n"
                "Click here immediately to claim your prize!\n"
                "Send your bank details to winner@lottery.com"
            )
            self.text_area.insert('1.0', placeholder_text)
            self.text_area.config(fg='gray')
    
    def load_file(self):
        """Charger un fichier texte"""
        file_path = filedialog.askopenfilename(
            title="Sélectionner un fichier email",
            filetypes=[
                ("Fichiers texte", "*.txt"),
                ("Tous les fichiers", "*.*")
            ]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                # Effacer le placeholder si nécessaire
                if self.text_area.get('1.0', 'end-1c').startswith('Collez ici'):
                    self.text_area.delete('1.0', 'end')
                    self.text_area.config(fg='#2c3e50')
                else:
                    self.text_area.delete('1.0', 'end')
                
                self.text_area.insert('1.0', content)
                self.status_label.config(text=f"✅ Fichier chargé: {file_path}")
                
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible de charger le fichier:\n{str(e)}")
    
    def clear_text(self):
        """Effacer la zone de texte"""
        self.text_area.delete('1.0', 'end')
        self.text_area.config(fg='gray')
        placeholder_text = (
            "Collez ici le contenu de l'email à analyser...\n\n"
            "Exemple:\n"
            "Congratulations! You have won $1,000,000!\n"
            "Click here immediately to claim your prize!\n"
            "Send your bank details to winner@lottery.com"
        )
        self.text_area.insert('1.0', placeholder_text)
        
        # Effacer les résultats
        self.result_area.config(state='normal')
        self.result_area.delete('1.0', 'end')
        self.result_area.config(state='disabled')
        
        self.status_label.config(text="✅ Zone de texte effacée")
    
    def analyze_email_threaded(self):
        """Analyser l'email dans un thread séparé"""
        threading.Thread(target=self.analyze_email, daemon=True).start()
    
    def analyze_email(self):
        """Analyser l'email"""
        # Obtenir le texte
        email_text = self.text_area.get('1.0', 'end-1c')
        
        # Vérifier si c'est le placeholder
        if email_text.startswith('Collez ici') or not email_text.strip():
            messagebox.showwarning("Attention", "Veuillez saisir un email à analyser.")
            return
        
        # Démarrer la barre de progression
        self.progress.start()
        self.analyze_button.config(state='disabled', text="🔄 Analyse en cours...")
        self.status_label.config(text="🔄 Analyse en cours...")
        
        try:
            # Analyser l'email
            result = self.predictor.analyze_email(email_text)
            formatted_result = self.predictor.format_analysis_result(result)
            
            # Afficher les résultats
            self.result_area.config(state='normal')
            self.result_area.delete('1.0', 'end')
            self.result_area.insert('1.0', formatted_result)
            self.result_area.config(state='disabled')
            
            # Mettre à jour le statut
            if 'error' not in result:
                consensus = result['consensus']
                result_text = "SPAM" if consensus['is_spam'] else "HAM"
                confidence = consensus['confidence']
                self.status_label.config(
                    text=f"✅ Analyse terminée: {result_text} ({confidence:.1%} confiance)"
                )
            else:
                self.status_label.config(text="❌ Erreur lors de l'analyse")
                
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'analyse:\n{str(e)}")
            self.status_label.config(text="❌ Erreur lors de l'analyse")
        
        finally:
            # Arrêter la barre de progression
            self.progress.stop()
            self.analyze_button.config(state='normal', text="🔍 ANALYSER L'EMAIL")
    
    def run(self):
        """Lancer l'interface graphique"""
        self.root.mainloop()

if __name__ == "__main__":
    app = SpamDetectorGUI()
    app.run()
