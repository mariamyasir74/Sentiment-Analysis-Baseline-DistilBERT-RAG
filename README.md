# Sentiment Analyzer
An end-to-end sentiment analysis project with RAG system trained with a fast baseline (TF-IDF + LR) and a fine-tuned transformer (DistilBERT). Includes a Streamlit demo and Dockerfile for easy sharing.

## Quickstart
1. Clone repo
2. Install packages
```bash
pip install -r requirements.txt
```
3. Train baseline:
```bash
python src/baseline.py
```

<img width="1090" height="526" alt="baseline" src="https://github.com/user-attachments/assets/c8cec855-5785-4866-b148-6549f8d672cf" />

4. Train transformer:
```bash
python src/transformers_train.py
```

<img width="1615" height="150" alt="transformers_train" src="https://github.com/user-attachments/assets/222ae642-b3ce-4a47-81b8-31d4ad51297b" />

5. Run demo:
```bash
streamlit run app/streamlit_app.py
```
<img width="766" height="777" alt="Sentiment Analyzer - Demo" src="https://github.com/user-attachments/assets/fe3eca55-7eb7-42a4-83dc-020aab7e48df" />

## Results

- Baseline accuracy:

<img width="464" height="733" alt="Baseline_negative_ex" src="https://github.com/user-attachments/assets/00960e4e-0cc6-4dc4-bcf7-1eafd3b371ca" />

---

 <img width="474" height="729" alt="Baseline_positive_ex" src="https://github.com/user-attachments/assets/66e2498c-ca41-42d2-9968-18695abc66e8" />

---

- DistilBERT accuracy:

<img width="312" height="744" alt="Transformer_negative_ex" src="https://github.com/user-attachments/assets/973de4d3-4b73-4dcf-a206-0a636c545501" />

---

<img width="311" height="744" alt="Transformer_positive_ex" src="https://github.com/user-attachments/assets/07ca5f3c-bb74-4fe0-9332-8b6d4fd049c9" />

---

## Sentiment Analyzer with RAG system
1. Upload .ipynp file on Google Colap and run the cells

<img width="1631" height="244" alt="Screenshot 2025-10-01 091007" src="https://github.com/user-attachments/assets/9f69f603-6af2-4f46-be14-680348eed1c2" />

---

<img width="377" height="206" alt="Screenshot 2025-10-01 091029" src="https://github.com/user-attachments/assets/756c95ca-a5a3-4bfe-bd5b-6c74f858ed08" />

3. Download the sentiment_model folder, .index and .pkl file
4. Run demo:
```bash
streamlit run app/RAG_streamlit_app.py
```
<img width="742" height="804" alt="Sentiment Analyser + RAG" src="https://github.com/user-attachments/assets/b9d08f2f-f138-4cd6-ad56-e67c0723c457" />

---

<img width="469" height="763" alt="transformer_RAG_ex" src="https://github.com/user-attachments/assets/cdddf633-3b24-44c4-a653-8858c6fc6587" />
