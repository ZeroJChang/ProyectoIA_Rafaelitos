import math
import pickle
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = defaultdict(float)
        self.word_probs = defaultdict(dict)  # Cambiado de lambda a dict
        self.vocabulary = set()
        self.class_word_counts = defaultdict(lambda: defaultdict(int))
    
    def train(self, train_set):
        """Entrena el clasificador Naïve Bayes"""
        # Contar documentos por clase
        class_counts = defaultdict(int)
        total_docs = len(train_set)
        
        # Contar palabras por clase
        for features, category in train_set:
            class_counts[category] += 1
            for word, present in features.items():
                if present:
                    self.class_word_counts[category][word] += 1
        
        # Calcular probabilidades a priori P(c)
        for category, count in class_counts.items():
            self.class_probs[category] = count / total_docs
        
        # Calcular probabilidades condicionales P(w|c) con suavizado Laplace
        for category in class_counts:
            total_words_in_class = sum(self.class_word_counts[category].values())
            vocab_size = len(self.vocabulary)
            
            for word in self.vocabulary:
                count = self.class_word_counts[category].get(word, 0)
                self.word_probs[word][category] = (count + 1) / (total_words_in_class + vocab_size)
    
    def predict(self, features):
        """Predice la categoría para un nuevo documento"""
        best_class = None
        max_log_prob = -float('inf')
        
        for category in self.class_probs:
            # Iniciar con log(P(c))
            log_prob = math.log(self.class_probs[category])
            
            # Sumar log(P(w|c)) para cada palabra presente
            for word, present in features.items():
                if present and word in self.word_probs:
                    log_prob += math.log(self.word_probs[word][category])
            
            if log_prob > max_log_prob:
                max_log_prob = log_prob
                best_class = category
        
        return best_class
    
    def save_model(self, filename):
        """Guarda el modelo entrenado de forma serializable"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'class_probs': dict(self.class_probs),
                'word_probs': {k: dict(v) for k, v in self.word_probs.items()},
                'vocabulary': list(self.vocabulary),
                'class_word_counts': {k: dict(v) for k, v in self.class_word_counts.items()}
            }, f)
    
    @classmethod
    def load_model(cls, filename):
        """Carga un modelo guardado"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        classifier = cls()
        classifier.class_probs = defaultdict(float, model_data['class_probs'])
        classifier.word_probs = {k: dict(v) for k, v in model_data['word_probs'].items()}
        classifier.vocabulary = set(model_data['vocabulary'])
        classifier.class_word_counts = defaultdict(lambda: defaultdict(int))
        for cat, counts in model_data['class_word_counts'].items():
            classifier.class_word_counts[cat] = defaultdict(int, counts)
        
        return classifier