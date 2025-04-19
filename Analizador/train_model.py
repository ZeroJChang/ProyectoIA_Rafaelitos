from collections import defaultdict
from naive_bayes import NaiveBayesClassifier
import csv
import os
import pickle
import numpy as np

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can't", "cannot",
    "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing",
    "don't", "down", "during", "each", "few", "for", "from", "further",
    "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he",
    "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its",
    "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't",
    "so", "some", "such", "than", "that", "that's", "the", "their",
    "theirs", "them", "themselves", "then", "there", "there's", "these",
    "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while", "who",
    "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't",
    "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}

def load_dataset(filename):
    """Carga mejorada con verificación de datos"""
    filepath = os.path.join("Analizador", "preprocessed", filename)
    
    dataset = []
    vocabulary = set()
    category_counts = defaultdict(int)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        vocab_list = next(reader)[:-1]  # Excluye categoría
        
        for row_num, row in enumerate(reader, 1):
            try:
                features = {word: int(count) for word, count in zip(vocab_list, row[:-1])}
                category = row[-1]
                
                # Verificar que la categoría es válida
                if category not in ["business", "entertainment", "politics", "sport", "tech"]:
                    raise ValueError(f"Categoría inválida en fila {row_num}: {category}")
                
                dataset.append((features, category))
                category_counts[category] += 1
                
                # Construir vocabulario
                for word, count in features.items():
                    if count > 0:
                        vocabulary.add(word)
                        
            except Exception as e:
                print(f"Error en fila {row_num}: {str(e)}")
                continue
    
    print("\nDistribución de categorías en los datos:")
    for cat, count in category_counts.items():
        print(f"- {cat}: {count} documentos ({count/len(dataset):.1%})")
    
    return dataset, vocabulary

def save_model(classifier, filename):
    """Guarda el modelo con verificación"""
    model_data = {
        'class_probs': classifier.class_probs,
        'word_probs': classifier.word_probs,
        'vocabulary': list(classifier.vocabulary),
        'stopwords': STOPWORDS
    }
    
    # Verificar estructura antes de guardar
    required_keys = ['class_probs', 'word_probs', 'vocabulary']
    for key in required_keys:
        if not model_data.get(key):
            raise ValueError(f"Modelo incompleto: falta {key}")
    
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)

def main():
    print("----------------ENTRENAMIENTO DEL MODELO----------------")
    
    try:
        # 1. Cargar datos con verificación
        print("\nCargando dataset de entrenamiento...")
        train_data, vocabulary = load_dataset('train_dataset.csv')
        
        if not train_data:
            raise ValueError("No hay datos de entrenamiento válidos")
        
        # 2. Entrenar modelo con validación
        print("\nEntrenando modelo Naïve Bayes...")
        classifier = NaiveBayesClassifier()
        classifier.train(train_data)
        
        # Validación rápida del modelo
        test_sample = {word: 1 for word in list(vocabulary)[:10]}
        print("\nPrueba de predicción rápida:")
        print(f"Predicción para muestra: {classifier.predict(test_sample)}")
        
        # 3. Guardar modelo
        model_path = os.path.join("Analizador", "bbc_classifier.pkl")
        save_model(classifier, model_path)
        
        print(f"\nModelo guardado en: {model_path}")
        print("\nProceso completado exitosamente!")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("Revisa los datos de entrenamiento")

if __name__ == '__main__':
    main()