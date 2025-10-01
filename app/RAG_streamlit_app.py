import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import faiss, pickle
from PIL import Image
import os

tokenizer = AutoTokenizer.from_pretrained("sentiment_model")
model = AutoModelForSequenceClassification.from_pretrained("sentiment_model")
clf = pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
gen_tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
index = faiss.read_index("faiss_index.index")
with open("passages.pkl", "rb") as f:
    train_texts = pickle.load(f)

def retrieve_passages(query, k=5):
    q_emb = embedder.encode([query])
    D, I = index.search(q_emb, k)
    return [train_texts[i] for i in I[0]]

def explain_sentiment(text, predicted_label, k=3):
    retrieved = retrieve_passages(text, k=k)
    retrieved_str = "\n- ".join(retrieved)
    prompt = f"""
    Text to analyze: "{text}"
    Predicted sentiment: {predicted_label}
    Retrieved examples:
    - {retrieved_str}
    Task: Write a clear explanation (1–2 sentences) about why the text is {predicted_label}. 
    Focus on emotional words, tone, and context. Do NOT just repeat the label.
    """
    inputs = gen_tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = gen_model.generate(**inputs, max_length=80)
    explanation = gen_tok.decode(outputs[0], skip_special_tokens=True)
    return {"retrieved": retrieved, "explanation": explanation}

st.title("Sentiment Analyzer")
image = Image.open("Social-Sentiment-Tracking.png")
st.image(image, width='stretch')
text = st.text_area("Enter text to analyze:")
if st.button("Predict"):
    out = clf(text)[0]
    pred = out['label']
    score = out['score']
    if pred == "LABEL_0":
        human_pred = 'Negative'
    elif pred == "LABEL_1":
        human_pred = 'Neutral'
    else:
        human_pred = 'Positive'
    st.write(f"**Prediction:** {human_pred} (confidence {score*100:.2f}%)")
    rag_output = explain_sentiment(text, pred)
    st.subheader("Retrieved evidence")
    for r in rag_output['retrieved']:
        st.write("-", r)
    st.subheader("Explanation")
    st.write(rag_output['explanation'])