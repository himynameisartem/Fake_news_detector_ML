# Fake News Detector

This project is a machine learning model designed to detect "fake news" with an accuracy of over 90%. It utilizes `TfidfVectorizer` for feature extraction from text data and a `PassiveAggressiveClassifier` for classification.

## Table of Contents
- [English Version](#english-version)
- [Russian Version](#russian-version)

---

## English Version

### Description
The primary goal of this project is to build a classic machine learning model using the `scikit-learn` library to accurately classify news articles as either "REAL" or "FAKE".

### Dataset
The model is trained on the `fake_news.csv` dataset included in this repository.

### Technologies Used
- Python
- scikit-learn
- pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

### Methodology

#### Feature Extraction: TfidfVectorizer
This project uses **Term Frequency-Inverse Document Frequency (TF-IDF)**.

`TfidfVectorizer` from `scikit-learn` converts a collection of raw documents into a matrix of TF-IDF features.

#### Classification Model: PassiveAggressiveClassifier

- It remains **passive** for correct classifications.
- It becomes **aggressive** when a miscalculation occurs, updating the model to correct for the error.

### Results
The model achieves an accuracy of over 90% in distinguishing between real and fake news. The results are visualized using a **confusion matrix** to provide a clear picture of the model's performance in terms of true positives, true negatives, false positives, and false negatives.

### How to Run
1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Install dependencies:**
    ```bash
    pip install numpy pandas scikit-learn matplotlib seaborn jupyterlab
    ```
3.  **Launch Jupyter Lab:**
    ```bash
    jupyter lab
    ```
4.  **Run the notebook:**
    Open and execute the cells in `Fake_news_detector.ipynb`.

---

## Russian Version

### Описание
Основная цель этого проекта — построить модель классического машинного обучения с использованием библиотеки `scikit-learn` для точной классификации новостных статей как «REAL» (настоящая) или «FAKE» (фейковая).

### Датасет
Модель обучается на датасете `fake_news.csv`, который находится в этом репозитории.

### Используемые технологии
- Python
- scikit-learn
- pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

### Методология

#### Извлечение признаков: TfidfVectorizer
В этом проекте используется **Term Frequency-Inverse Document Frequency (TF-IDF)**.

`TfidfVectorizer` из `scikit-learn` преобразует коллекцию необработанных документов в матрицу TF-IDF признаков.

#### Модель классификации: PassiveAggressiveClassifier

- Он остается **пассивным** при правильных классификациях.
- Он становится **агрессивным** при ошибках, обновляя модель для их исправления.

### Результаты
Модель достигает точности более 90% в различении настоящих и фейковых новостей. Результаты визуализированы с помощью **матрицы ошибок (confusion matrix)**, которая дает ясное представление о производительности модели (истинно-положительные, истинно-отрицательные, ложно-положительные и ложно-отрицательные срабатывания).

### Как запустить
1.  **Склонируйте репозиторий:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Установите зависимости:**
    ```bash
    pip install numpy pandas scikit-learn matplotlib seaborn jupyterlab
    ```
3.  **Запустите Jupyter Lab:**
    ```bash
    jupyter lab
    ```
4.  **Запустите ноутбук:**
    Откройте и выполните ячейки в `Fake_news_detector.ipynb`.
