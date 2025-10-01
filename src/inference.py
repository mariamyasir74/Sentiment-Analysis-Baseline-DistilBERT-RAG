import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

class BaselinePredictor:
    def __init__(self, path=r'D:\Sentiment Analysis project\src\models\baseline.joblib'):
        self.pipe = joblib.load(path)
    def predict(self, texts):
        return self.pipe.predict_proba(texts)

class TransformerPredictor:
    def __init__(self, model_dir=r'D:\Sentiment Analysis project\src\models\distilbert-sst'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
    def predict(self, texts):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probs.numpy()
