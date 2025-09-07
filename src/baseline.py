import joblib
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

def train_baseline(output_path='models/baseline.joblib'):
    ds = load_dataset('imdb')
    train_texts = [x['text'] for x in ds['train']]
    train_labels = [x['label'] for x in ds['train']]
    test_texts = [x['text'] for x in ds['test']]
    test_labels = [x['label'] for x in ds['test']]
    pipe = Pipeline([('tfidf', TfidfVectorizer(max_features=40000, ngram_range=(1, 2))),
                    ('clf', LogisticRegression(max_iter=1000))])
    pipe.fit(train_texts, train_labels)
    preds = pipe.predict(test_texts)
    print(classification_report(test_labels, preds))
    joblib.dump(pipe, output_path)
    print('Baseline model saved to ', output_path)

if __name__ == '__main__':
    train_baseline()
