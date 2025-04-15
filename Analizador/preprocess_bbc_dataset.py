import os
import re
import csv
import glob
import random
from collections import Counter, defaultdict

# Configuración de rutas
NEWS_PATH = os.path.join("DataSet", "BBC News Summary", "BBC News Summary", "News Articles")
SUMMARIES_PATH = os.path.join("DataSet", "BBC News Summary", "BBC News Summary", "Summaries")
CATEGORIES = ["business", "entertainment", "politics", "sport", "tech"]
TEST_SIZE = 0.2  # 20% para prueba
SEED = 42
random.seed(SEED)

# Lista completa de stopwords
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

def preprocess_text(text):
    """Preprocesamiento completo del texto"""
    # Convertir a minúsculas
    text = text.lower()
    
    # Eliminar caracteres especiales y números
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenización y filtrado
    words = text.split()
    return [word for word in words 
            if word not in STOPWORDS  # Eliminar stopwords
            and len(word) > 2  # Filtrar palabras cortas
            and word.isalpha()]  # Solo palabras alfabéticas

def process_category_files(base_path, categories):
    """Procesa todos los archivos por categoría"""
    category_data = {category: [] for category in categories}
    
    for category in categories:
        search_path = os.path.join(base_path, category, "**/*.txt")
        
        for file_path in glob.glob(search_path, recursive=True):
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                    content = file.read().strip()
                    if content:
                        tokens = preprocess_text(content)
                        category_data[category].extend(tokens)
            except Exception as e:
                print(f"Error procesando {file_path}: {str(e)}")
    
    return category_data

def generate_keywords_analysis(category_data, top_n=150):
    """Genera el análisis de palabras clave por categoría"""
    keywords_output = []
    
    for category, words in category_data.items():
        word_counts = Counter(words)
        for word, count in word_counts.most_common(top_n):
            keywords_output.append([category, word, count])
    
    return keywords_output

def prepare_ml_datasets(category_data, vocab_size=2000):
    """Prepara los datasets para machine learning"""
    # Crear documentos artificiales (100 palabras por documento)
    documents = []
    for category, words in category_data.items():
        documents.extend([(words[i:i+100], category) for i in range(0, len(words), 100)])
    
    # Construir vocabulario
    all_words = [word for words, _ in documents for word in words]
    vocabulary = [word for word, _ in Counter(all_words).most_common(vocab_size)]
    
    # Crear vectores de características
    features = []
    vocab_set = set(vocabulary)
    for words, category in documents:
        features.append(({word: 1 if word in vocab_set else 0 for word in vocabulary}, category))
    
    # Dividir en train/test manteniendo proporción por categoría
    random.shuffle(features)
    split_idx = int(len(features) * (1 - TEST_SIZE))
    train_set, test_set = features[:split_idx], features[split_idx:]
    
    return vocabulary, train_set, test_set

def save_keywords_analysis(keywords_output):
    """Guarda el análisis de palabras clave"""
    with open("palabras_por_categoria.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Categoría", "Palabra", "Frecuencia"])
        writer.writerows(keywords_output)

def save_ml_datasets(vocabulary, train_set, test_set):
    """Guarda los datasets para ML"""
    # Crear directorio si no existe
    os.makedirs(os.path.join("Analizador", "preprocessed"), exist_ok=True)
    
    # Guardar train dataset
    train_path = os.path.join("Analizador", "preprocessed", "train_dataset.csv")
    with open(train_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(vocabulary + ["category"])
        for features, category in train_set:
            writer.writerow(list(features.values()) + [category])
    
    # Guardar test dataset
    test_path = os.path.join("Analizador", "preprocessed", "test_dataset.csv")
    with open(test_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(vocabulary + ["category"])
        for features, category in test_set:
            writer.writerow(list(features.values()) + [category])
    
    # Guardar vocabulario
    vocab_path = os.path.join("Analizador", "preprocessed", "vocabulary.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write("\n".join(vocabulary))

def main():
    print("=== PROCESAMIENTO DE DATOS BBC NEWS ===")
    print("Objetivos:")
    print("1. Generar análisis de palabras clave")
    print("2. Preparar datasets para entrenamiento del modelo\n")
    
    # Paso 1: Procesar archivos
    print("Procesando News Articles...")
    news_data = process_category_files(NEWS_PATH, CATEGORIES)
    
    print("Procesando Summaries...")
    summaries_data = process_category_files(SUMMARIES_PATH, CATEGORIES)
    
    # Combinar datos de ambas fuentes
    combined_data = {category: news_data[category] + summaries_data[category] 
                    for category in CATEGORIES}
    
    # Paso 2: Generar análisis de palabras clave
    print("\nGenerando análisis de palabras clave...")
    keywords_output = generate_keywords_analysis(combined_data)
    save_keywords_analysis(keywords_output)
    
    # Paso 3: Preparar datasets para ML
    print("Preparando datasets para machine learning...")
    vocabulary, train_set, test_set = prepare_ml_datasets(combined_data)
    save_ml_datasets(vocabulary, train_set, test_set)
    
    # Resultados
    print("\n=== ARCHIVOS GENERADOS ===")
    print("1. palabras_por_categoria.csv")
    print("   - Top 150 palabras por categoría")
    print("2. Analizador/preprocessed/train_dataset.csv")
    print(f"   - {len(train_set)} documentos de entrenamiento")
    print("3. Analizador/preprocessed/test_dataset.csv")
    print(f"   - {len(test_set)} documentos de prueba")
    print("4. Analizador/preprocessed/vocabulary.txt")
    print(f"   - {len(vocabulary)} palabras en el vocabulario")
    print("\nProceso completado exitosamente!")

if __name__ == "__main__":
    main()