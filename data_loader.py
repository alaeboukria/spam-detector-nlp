import pandas as pd
import requests
import os
import zipfile
from io import BytesIO


class DataLoader:
    def __init__(self):
        self.data_dir = "data"
        self.dataset_url = "https://github.com/MWiechmann/enron_spam_data/raw/master/enron_spam_data.zip"
        self.dataset_path = os.path.join(self.data_dir, "enron_spam_data.csv")

    def ensure_data_dir(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def download_dataset(self):
        print("📥 Téléchargement du dataset Enron Spam (~33 k emails)...")
        try:
            resp = requests.get(self.dataset_url)
            resp.raise_for_status()
            z = zipfile.ZipFile(BytesIO(resp.content))
            # Le CSV comprimé s'appelle généralement enron_spam_data.csv
            z.extractall(self.data_dir)
            df = pd.read_csv(self.dataset_path)
            df = df.rename(columns={
                'Spam/Ham': 'label',
                'Message': 'message'
            })
            df['label'] = df['label'].map({'ham': 0, 'spam': 1})
            print(f"✅ Dataset téléchargé : {len(df)} messages")
            print(f"   - HAM : {len(df[df['label'] == 0])}")
            print(f"   - SPAM : {len(df[df['label'] == 1])}")
            return df
        except Exception as e:
            print(f"❌ Erreur téléchargement Enron Spam : {e}")
            return self.create_fallback_dataset()

    def load_dataset(self):
        if os.path.exists(self.dataset_path):
            print("📂 Chargement dataset Enron existant...")
            df = pd.read_csv(self.dataset_path)
            df['label'] = df['label'].map({'ham': 0, 'spam': 1})
            return df
        else:
            return self.download_dataset()
