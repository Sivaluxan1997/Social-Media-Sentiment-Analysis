# Sentiment Analysis on Social Media Posts

This project performs **Sentiment Analysis** on social media posts to classify emotions or opinions as **Positive**, **Negative**, or **Neutral**.  
It demonstrates the full machine learning workflow — from data preprocessing and visualization to model training and evaluation — all inside a single Jupyter Notebook.

---

##  Overview

Sentiment analysis is a Natural Language Processing (NLP) task used to determine whether text expresses positive, negative, or neutral emotion.  
This notebook builds a **machine learning pipeline** that processes text data and predicts sentiment labels.

---

## 📊 Dataset

The dataset is loaded using:
```python
import pandas as pd
data = pd.read_csv("data.csv")
```

Each record represents a **social media post** with attributes such as:
- Text (post content)
- Sentiment label (Positive / Negative / Neutral)
- Metadata (hashtags, time, user, etc.)

Example:
| Text | Sentiment | Platform | Likes | Shares |
|------|------------|-----------|--------|---------|
| Enjoying a beautiful day at the park! | Positive | Twitter | 15 | 30 |
| Traffic was terrible this morning. | Negative | Twitter | 5 | 10 |

---

##  Data Preprocessing

Main preprocessing steps:
- Lowercasing and punctuation removal  
- Stopword removal using **NLTK**  
- Tokenization and optional lemmatization  
- Text vectorization using `CountVectorizer` or `TfidfVectorizer`  
- Splitting dataset into training and testing sets  

Example snippet:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['Text'])
```

---

## Exploratory Analysis

The notebook includes visual exploration of:
- Sentiment distribution  
- Word frequency per sentiment  
- Word clouds for positive, neutral, and negative posts  

Tools used:
- `matplotlib`
- `seaborn`
- `wordcloud`

Example:
```python
sns.countplot(data=data, x='Sentiment')
```

---

## Model Training

A **Logistic Regression** classifier was trained to predict sentiments.

Example:
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
```

The simplicity and interpretability of Logistic Regression make it ideal for baseline text classification.

---

## 📈 Evaluation

Performance was evaluated using the `classification_report`:
```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

Metrics include:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

Confusion matrices and accuracy plots are used for visual inspection.

---

## 🚀 Future Improvements

- Experiment with deep learning (e.g., LSTM, BERT)
- Hyperparameter optimization
- Cross-platform sentiment comparison (Twitter vs Instagram)
- Model deployment as an API or Streamlit dashboard

---

