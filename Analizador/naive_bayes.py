import math
from collections import defaultdict
import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}
        self.word_probs = {}
        self.vocabulary = set()
        self.class_word_counts = {}
        self.class_total_words = {}
    
    def train(self, train_set):
        """Entrenamiento mejorado con verificación de datos"""
        # Reinicializar estructuras
        self.class_probs = {}
        self.word_probs = {}
        self.class_word_counts = {}
        self.class_total_words = {}
        
        # Contar documentos por clase
        class_counts = defaultdict(int)
        total_docs = len(train_set)
        
        # Verificación inicial
        if total_docs == 0:
            raise ValueError("El conjunto de entrenamiento está vacío")
        
        # Primera pasada: contar documentos y palabras
        for features, category in train_set:
            class_counts[category] += 1
            
            if category not in self.class_word_counts:
                self.class_word_counts[category] = defaultdict(int)
                self.class_total_words[category] = 0
            
            for word, count in features.items():
                if count > 0:
                    self.class_word_counts[category][word] += count
                    self.class_total_words[category] += count
                    self.vocabulary.add(word)
        
        # Calcular probabilidades a priori P(c)
        for category, count in class_counts.items():
            self.class_probs[category] = count / total_docs
        
        # Calcular probabilidades condicionales P(w|c) con suavizado Laplace
        vocab_size = len(self.vocabulary)
        for category in class_counts:
            self.word_probs[category] = {}
            total_words = self.class_total_words[category]
            
            for word in self.vocabulary:
                count = self.class_word_counts[category].get(word, 0)
                self.word_probs[category][word] = (count + 1) / (total_words + vocab_size)
    
    def predict(self, features):
        """Predicción con verificación de entradas"""
        if not self.class_probs:
            raise ValueError("Modelo no entrenado")
            
        best_class = None
        max_log_prob = -np.inf
        
        for category in self.class_probs:
            try:
                log_prob = math.log(self.class_probs[category])
                
                for word, count in features.items():
                    if count > 0 and word in self.vocabulary:
                        prob = self.word_probs[category].get(word, 1e-10)  # Evitar ceros
                        log_prob += math.log(prob) * count
                
                if log_prob > max_log_prob or best_class is None:
                    max_log_prob = log_prob
                    best_class = category
                    
            except Exception as e:
                print(f"Error en categoría {category}: {str(e)}")
                continue
        
        return best_class