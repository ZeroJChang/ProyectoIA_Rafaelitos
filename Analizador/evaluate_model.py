from collections import defaultdict
import pickle
import os
import csv
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset(filename):
    """Carga el dataset de prueba"""
    filepath = os.path.join("Analizador", "preprocessed", filename)
    
    features = []
    true_labels = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        vocabulary = next(reader)[:-1]  # Excluye la columna de categoría
        
        for row in reader:
            features.append([int(count) for count in row[:-1]])
            true_labels.append(row[-1])
    
    return features, true_labels, vocabulary

def predict_with_model(features, vocabulary, model_data):
    """Realiza predicciones usando el modelo cargado"""
    pred_labels = []
    class_probs = model_data['class_probs']
    word_probs = model_data['word_probs']
    
    for feature_row in features:
        features_dict = {word: count for word, count in zip(vocabulary, feature_row)}  
        
        best_class = None
        max_log_prob = -float('inf')
        
        for category in class_probs:
            log_prob = math.log(class_probs[category])
            
            for word, count in features_dict.items():
                if count > 0 and word in word_probs.get(category, {}):
                    log_prob += math.log(word_probs[category][word]) * count
            
            if log_prob > max_log_prob:
                max_log_prob = log_prob
                best_class = category
        
        pred_labels.append(best_class)
    
    return pred_labels


def print_metrics(true_labels, pred_labels, categories):
    """Calcula e imprime métricas de evaluación"""
    # Métricas generales
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    
    print("\n----------------MÉTRICAS GENERALES----------------")
    print(f"Precisión: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1-Score: {f1:.2%}")
    
    # Métricas por categoría
    print("\n----------------MÉTRICAS POR CATEGORÍA----------------")
    print(f"{'Categoría':<15} {'Precisión':<10} {'Recall':<10} {'F1-Score':<10}")
    for category in categories:
        precision = precision_score(true_labels, pred_labels, labels=[category], average='micro')
        recall = recall_score(true_labels, pred_labels, labels=[category], average='micro')
        f1 = f1_score(true_labels, pred_labels, labels=[category], average='micro')
        print(f"{category:<15} {precision:<10.2%} {recall:<10.2%} {f1:<10.2%}")
    
    # Matriz de confusión
    conf_matrix = confusion_matrix(true_labels, pred_labels, labels=categories)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=categories, yticklabels=categories)
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.savefig('Analizador/preprocessed/confusion_matrix.png')
    print("\nMatriz de confusión guardada en 'Analizador/preprocessed/confusion_matrix.png'")
    
def evaluate_model(true_labels, pred_labels, categories):
    """Evaluación mejorada con manejo de casos extremos"""
    # Configurar zero_division para evitar warnings
    precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
    
    print("\n----------------MÉTRICAS GENERALES----------------")
    print(f"Precisión: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1-Score: {f1:.2%}")
    
    print("\n----------------MÉTRICAS POR CATEGORÍA----------------")
    print(f"{'Categoría':<15} {'Precisión':<10} {'Recall':<10} {'F1-Score':<10} {'Ejemplos':<10}")
    
    for category in categories:
        # Filtramos solo las predicciones para esta categoría
        y_true_cat = [1 if y == category else 0 for y in true_labels]
        y_pred_cat = [1 if y == category else 0 for y in pred_labels]
        
        tp = sum(1 for t, p in zip(y_true_cat, y_pred_cat) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true_cat, y_pred_cat) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true_cat, y_pred_cat) if t == 1 and p == 0)
        
        precision_cat = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_cat = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_cat = 2 * (precision_cat * recall_cat) / (precision_cat + recall_cat) if (precision_cat + recall_cat) > 0 else 0
        count = sum(1 for y in true_labels if y == category)
        
        print(f"{category:<15} {precision_cat:<10.2%} {recall_cat:<10.2%} {f1_cat:<10.2%} {count:<10}")

def main():
    print("----------------EVALUACIÓN DEL MODELO----------------")
    
    try:
        # 1. Cargar modelo con verificación
        model_path = os.path.join("Analizador", "bbc_classifier.pkl")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
            # Verificar estructura del modelo
            required_keys = ['class_probs', 'word_probs', 'vocabulary']
            for key in required_keys:
                if key not in model_data:
                    raise ValueError(f"Modelo corrupto: falta {key}")
        
        # 2. Cargar datos de prueba
        test_features, true_labels, vocabulary = load_dataset('test_dataset.csv')
        
        # 3. Predecir y evaluar
        pred_labels = predict_with_model(test_features, vocabulary, model_data)
        categories = sorted(list(set(true_labels)))
        
        evaluate_model(true_labels, pred_labels, categories)
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")

if __name__ == '__main__':
    import math
    main()