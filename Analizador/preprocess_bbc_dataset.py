import os
import re
import csv
import glob
from collections import Counter

# Configuración de rutas
NEWS_PATH = r"C:\IA\ProyectoIA_Rafaelitos\DataSet\BBC News Summary\BBC News Summary\News Articles"
SUMMARIES_PATH = r"C:\IA\ProyectoIA_Rafaelitos\DataSet\BBC News Summary\BBC News Summary\Summaries"
CATEGORIES = ["business", "entertainment", "politics", "sport", "tech"]

# Stopwords personalizadas
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
    """Preprocesamiento robusto del texto"""
    # Normalización a minúsculas
    text = text.lower()
    
    # Eliminación de caracteres no alfabéticos
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenización y filtrado
    words = text.split()
    return [word for word in words 
            if word not in STOPWORDS  # Elimina stopwords
            and len(word) > 2  # Filtra palabras cortas
            and word.isalpha()]  # Solo palabras alfabéticas

def process_category(category_path):
    """Procesa todos los archivos de una categoría"""
    category_words = []
    
    for file_path in glob.glob(os.path.join(category_path, "**/*.txt"), recursive=True):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                content = file.read().strip()
                if content:
                    tokens = preprocess_text(content)
                    category_words.extend(tokens)
        except Exception as e:
            print(f"Error procesando {file_path}: {str(e)}")
    
    return category_words

def analyze_categories(base_paths, categories, top_n=150):
    """Analiza las palabras más frecuentes por categoría"""
    category_analysis = {}
    
    for category in categories:
        print(f"Procesando categoría: {category}")
        all_words = []
        
        # Procesar ambas fuentes (artículos y resúmenes)
        for base_path in base_paths:
            category_path = os.path.join(base_path, category)
            if os.path.exists(category_path):
                all_words.extend(process_category(category_path))
        
        # Contar palabras y seleccionar las más frecuentes
        word_counts = Counter(all_words)
        top_words = word_counts.most_common(top_n)
        category_analysis[category] = top_words
    
    return category_analysis

def save_to_csv(category_analysis, output_file):
    """Guarda el análisis en un archivo CSV"""
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Categoría", "Palabra", "Frecuencia"])
        
        for category, words in category_analysis.items():
            for word, count in words:
                writer.writerow([category, word, count])
    
    print(f"\nResultados guardados en: {output_file}")

if __name__ == "__main__":
    print("Iniciando análisis de palabras por categoría...")
    print("(Combinando artículos y resúmenes)\n")
    
    # Analizar ambas fuentes de datos
    base_paths = [NEWS_PATH, SUMMARIES_PATH]
    category_analysis = analyze_categories(base_paths, CATEGORIES)
    
    # Guardar resultados
    save_to_csv(category_analysis, "palabras_por_categoria.csv")
    
    print("\nProceso completado!")