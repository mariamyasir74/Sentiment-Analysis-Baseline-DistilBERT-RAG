import streamlit as st
import sys
import os
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inference import BaselinePredictor, TransformerPredictor
import numpy as np
from lime.lime_text import LimeTextExplainer
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sentiment Analyzer", layout='centered')
st.title('Sentiment Analyzer')
image = Image.open(r"D:\Sentiment Analysis project\cf8e9670-4661-4e8e-b58f-101a615dfca7.png")
st.image(image, use_container_width=True)
option = st.radio('Model', ['Baseline (TF-IDF)', 'Transformer (DistilBERT)'])
text = st.text_area('Enter text to analyze', height=150)
if st.button('Predict') and text.strip():
    if option.startswith('Baseline'):
        model = BaselinePredictor()
        pred = model.predict([text])[0]
        pos_prob = pred[1]
        explainer = LimeTextExplainer(class_names=["negative", "positive"])
        exp = explainer.explain_instance(text, model.predict, num_features=10)
        lime_html = exp.as_html()
    else:
        model = TransformerPredictor()
        pred = model.predict([text])[0]
        pos_prob = float(pred[1])
        st.write("### SHAP Explanation (fast mode)")
        explainer = shap.Explainer(model.predict, masker=shap.maskers.Text(), algorithm="partition")
        shap_values = explainer([text], max_evals=500)
        shap.plots.text(shap_values[0], display=False)
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        st.pyplot(fig, bbox_inches="tight")
    st.write('### Prediction')
    label = 'Positive' if pos_prob >= 0.5 else 'Negative'
    st.write(label)
    st.write('---')
    st.write('**Model output (raw probabilities)**')
    st.json({'negative': float(1 - pos_prob), 'positive': pos_prob})