
import os
import re
import csv

# Ruta del dataset (ajusta esto si tu dataset está en otra ubicación)
BASE_PATH = "./BBC News Summary/News Articles"

# Stopwords básicas en inglés
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with"
}

def simple_preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    return [word for word in words if word not in STOPWORDS]

def load_dataset(base_path):
    dataset = []
    for category in os.listdir(base_path):
        category_path = os.path.join(base_path, category)
        if os.path.isdir(category_path):
            for filename in os.listdir(category_path):
                file_path = os.path.join(category_path, filename)
                with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                    content = file.read().strip()
                    if content:
                        tokens = simple_preprocess(content)
                        dataset.append((" ".join(tokens), category))
    return dataset

def save_to_csv(data, output_path="preprocessed_bbc_news.csv"):
    with open(output_path, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["text", "category"])
        writer.writerows(data)

if __name__ == "__main__":
    print("Cargando y preprocesando dataset...")
    dataset = load_dataset(BASE_PATH)
    print(f"Total de muestras: {len(dataset)}")
    save_to_csv(dataset)
    print("Dataset guardado como 'preprocessed_bbc_news.csv'")
