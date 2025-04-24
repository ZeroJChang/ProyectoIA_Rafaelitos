# Clasificador de Noticias con Naïve Bayes – Proyecto Rafaelitos

Este proyecto implementa un sistema completo para clasificar noticias usando el algoritmo Naïve Bayes desde cero. Incluye un backend en Python (Flask), un frontend en React, y un motor de inferencia.

---

## Uso

### 1. El usuario ingresa una noticia en el frontend (web).
### 2. La noticia se envía al backend vía `POST /classify`.
### 3. El backend la preprocesa y la evalúa con el modelo Naïve Bayes.
### 4. Devuelve:
- Categoría más probable
- Porcentaje de certeza
- (Opcional) Segunda categoría con alta probabilidad

---

##  Arquitectura
![image](https://github.com/user-attachments/assets/5c5e455c-8765-4421-9bf6-0a8c072edd3c)


---

##  Descripción de archivos y carpetas

###  Analizador
- `api.py`: Servidor Flask, recibe texto y devuelve la predicción.
- `naive_bayes.py`: Implementación propia del clasificador Naïve Bayes.
- `train_model.py`: Entrena y guarda el modelo (`bbc_classifier.pkl`).
- `evaluate_model.py`: Evalúa el rendimiento del modelo.
- `preprocess_bbc_dataset.py`: Limpia y organiza el dataset original.
- `bbc_classifier.pkl`: Modelo entrenado con probabilidades.
- `preprocessed/`: Contiene `train_dataset.csv`, `test_dataset.csv`, y `vocabulary.txt`.

###  Frontend
- `App.js`: Componente principal en React.
- `index.js`: Punto de entrada.
- `App.css`: Estilos generales.
- `assets/`: Imagen grupal del proyecto.

---

##  Tecnologías utilizadas

- **Python 3.13**
- **Flask + Flask-CORS**
- **React.js**
- **Naïve Bayes desde cero (sin sklearn)**
- **CSV y Pickle** para datos y modelos

---
##  Requisitos técnicos

- **Node.js:** Requerido para ejecutar y desarrollar el frontend. Recomendado instalar la última versión estable.
- **npm:** Gestor de paquetes de Node.js. Se instala junto con Node.js.
- **React:** Biblioteca principal utilizada para construir la interfaz.
- **HTML, CSS, JavaScript moderno (ES6+):** Para la estructura, diseño y funcionalidad del frontend.

##  Evaluación del Modelo
![image](https://github.com/user-attachments/assets/f2dc741c-26f9-47ae-a63a-08cc2ebec4ca)

---

##  Dataset

Se utilizó el corpus **BBC News Summary**, con 5 categorías:
- `business`, `entertainment`, `politics`, `sport`, `tech`

El preprocesamiento incluyó:
- Normalización a minúsculas
- Eliminación de puntuación
- Filtro de stopwords y tokens no alfabéticos

---

##  Instalación

###  Requisitos previos
- Python 3.8 o superior
- Node.js y npm

###  Backend
```bash
cd Analizador
pip install flask flask-cors
```

 Luego de ya tener los requisitos instalados se necesita descar el repositorio 

 Se accede a la ruta del proyecto 

###  Iniciar Motor Naïve Bayes (Backend)

![image](https://github.com/user-attachments/assets/11dcd23f-21d3-4456-9a54-a2acbec613e4)

 ###  Iniciar la pagina web (Frontend)

![image](https://github.com/user-attachments/assets/e3c2ceaa-77dd-48ca-a94a-322a5dc21459)




