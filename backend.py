import numpy as np
import pickle
import re
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory 

class Preprocessing:
    def __init__(self):
        # 1. Kamus Gaul Gojek
        self.norm_dict = {
            'apk': 'aplikasi', 'app': 'aplikasi', 'bgt': 'banget', 'bgtt': 'banget',
            'tdk': 'tidak', 'gak': 'tidak', 'gk': 'tidak', 'ga': 'tidak', 'enggak': 'tidak',
            'g': 'tidak', 'gd': 'tidak ada', 'gbisa': 'tidak bisa', 'nda': 'tidak',
            'sy': 'saya', 'jg': 'juga', 'jd': 'jadi', 'krn': 'karena', 'karna': 'karena', 
            'kalo': 'kalau', 'dgn': 'dengan', 'yg': 'yang', 'udah': 'sudah', 'udh': 'sudah', 
            'ud': 'sudah', 'blm': 'belum', 'driver': 'pengemudi', 'drivernya': 'pengemudinya',
            'ojol': 'ojek online', 'gopay': 'saldo', 'min': 'admin', 'nyasar': 'tersesat', 
            'lemot': 'lambat', 'lola': 'lambat', 'error': 'rusak', 'update': 'perbarui', 
            'cpt': 'cepat', 'bgs': 'bagus', 'dr': 'dari', 'tp': 'tapi', 'sm': 'sama', 
            'ny': 'nya', 'pdhl': 'padahal', 'lgi': 'lagi', 'kek': 'seperti', 'cmn': 'cuma',
            'hr': 'hari', 'ni': 'ini'
        }
        
        # 2. Stopwords (KECUALI 'tidak' dan 'belum')
        self.stopwords = [
            "dan", "di", "ke", "dari", "yang", "pada", "untuk", 
            "dengan", "ini", "itu", "saya", "aku", "nya", "sebagai"
        ]
        
        # 3. Sastrawi Stemmer
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+|@\S+", '', text)
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        words = text.split()
        words = [self.norm_dict.get(w, w) for w in words]
        words = [w for w in words if w not in self.stopwords]
        
        clean_sentence = " ".join(words)
        final_text = self.stemmer.stem(clean_sentence)
        
        return final_text

class LSTM_Model:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.max_len = 50 
        self.load_resources()

    def load_resources(self):
        try:
            self.model = load_model('model_lstm.h5')
            with open('tokenizer.pickle', 'rb') as handle:
                self.tokenizer = pickle.load(handle)
            return True
        except Exception as e:
            print(f"Error loading model/tokenizer: {e}")
            return False

    def predict_single(self, text):
        """Prediksi Biner untuk satu ulasan beserta persentase Confidence"""
        if not self.model or not self.tokenizer:
            return "Error", 0.0

        clean_txt = Preprocessing().clean_text(text)
        seq = self.tokenizer.texts_to_sequences([clean_txt])
        padded = pad_sequences(seq, maxlen=self.max_len)
        
        pred = self.model.predict(padded)
        
        # Menghitung persentase keyakinan (Confidence Score)
        if pred.shape[-1] == 1:
            prob = float(pred[0][0])
            label_index = int(prob > 0.5)
            # Jika tebakannya positif (1), probabilitas aslinya. Jika negatif (0), sisa dari 100%
            confidence = prob if label_index == 1 else (1.0 - prob)
        else:
            label_index = int(np.argmax(pred))
            confidence = float(np.max(pred))
        
        labels = ["Negatif", "Positif"] 
        # Mengembalikan 2 nilai sekaligus: Label dan Persentase
        return labels[label_index], confidence

    def evaluate_model(self, df, text_col, label_col):
        """Evaluasi Dataset Biner dan Menghitung Total Sentimen"""
        if not self.model or not self.tokenizer:
            return None

        if text_col.lower() == 'clean_text':
            clean_texts = df[text_col].astype(str)
        else:
            clean_texts = df[text_col].apply(Preprocessing().clean_text)
            
        seq = self.tokenizer.texts_to_sequences(clean_texts)
        X = pad_sequences(seq, maxlen=self.max_len)
        
        label_map = {
            'negatif': 0, 'negative': 0, '0': 0, '0.0': 0,
            'positif': 1, 'positive': 1, '1': 1, '1.0': 1
        }
        
        y_true_mapped = df[label_col].astype(str).str.lower().str.strip().map(label_map)
        valid_indices = y_true_mapped.notna()
        X = X[valid_indices]
        y_true = y_true_mapped[valid_indices].astype(int)
        
        if len(y_true) == 0:
            return None
        
        y_pred_prob = self.model.predict(X)
        if y_pred_prob.shape[-1] == 1:
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        else:
            y_pred = np.argmax(y_pred_prob, axis=1)
        
        # DI SINI KUNCI PENYELESAIAN ERRORNYA:
        # Kita memasukkan total_positif dan total_negatif ke dalam metrik
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0),
            'total_positif': int(np.sum(y_pred == 1)), # Menghitung jumlah Prediksi Positif
            'total_negatif': int(np.sum(y_pred == 0))  # Menghitung jumlah Prediksi Negatif
        }
        return metrics