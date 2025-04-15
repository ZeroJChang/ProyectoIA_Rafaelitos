from naive_bayes import NaiveBayesClassifier
import csv
import os

def load_dataset(filename):
    """Carga el dataset preprocesado con rutas relativas correctas"""
    # Construir ruta al archivo en la carpeta preprocessed
    filepath = os.path.join("Analizador", "preprocessed", filename)
    
    dataset = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        vocabulary = next(reader)[:-1]  # Excluye la columna de categoría
        
        for row in reader:
            features = {word: int(count) for word, count in zip(vocabulary, row[:-1])}
            category = row[-1]
            dataset.append((features, category))
    
    return dataset

def main():
    #1. Cargagamos los datos datos
    try:
        train_data = load_dataset('train_dataset.csv')
    except FileNotFoundError:
        print("\nERROR: No se encontró train_dataset.csv")
        return
    
    # 2. Entrenar modelo
    print("\nEntrenando modelo Naïve Bayes...")
    classifier = NaiveBayesClassifier()
    classifier.train(train_data)
    
    # 3. Guardar modelo (en la carpeta Analizador)
    model_path = os.path.join("Analizador", "bbc_classifier.pkl")
    classifier.save_model(model_path)
    print(f"\nModelo entrenado y guardado como '{model_path}'")
    print("\nProceso completado exitosamente!")

if __name__ == '__main__':
    print("-------- ENTRENAMIENTO DEL MODELO -------")
    main()