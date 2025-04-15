from naive_bayes import NaiveBayesClassifier
import csv
from collections import defaultdict

def load_dataset(filename):
    """Carga el dataset de prueba"""
    dataset = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  
        for row in reader:
            features = {word: int(count) for word, count in zip(vocabulary, row[:-1])}
            category = row[-1]
            dataset.append((features, category))
    return dataset

def evaluate(classifier, test_data):
    """Evalúa el rendimiento del modelo"""
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    correct = 0
    
    for features, true_category in test_data:
        predicted = classifier.predict(features)
        confusion_matrix[true_category][predicted] += 1
        if predicted == true_category:
            correct += 1
    
    accuracy = correct / len(test_data)
    print(f"\nExactitud: {accuracy:.2%}")
    
    # Métricas por categoría
    print("\nMétricas por categoría:")
    for category in classifier.class_probs:
        tp = confusion_matrix[category][category]
        fp = sum(confusion_matrix[other][category] for other in classifier.class_probs if other != category)
        fn = sum(confusion_matrix[category][other] for other in classifier.class_probs if other != category)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{category.upper():<12} Precisión: {precision:.2%} | Recall: {recall:.2%} | F1: {f1:.2%}")

def main():
    # 1. Cargar modelo
    classifier = NaiveBayesClassifier.load_model('bbc_classifier.pkl')
    
    # 2. Cargar datos de prueba
    test_data = load_dataset('test_dataset.csv')
    
    # 3. Evaluar
    print("Evaluando modelo...")
    evaluate(classifier, test_data)

if __name__ == '__main__':
    main()