# Sentiment Analyzer
An end-to-end sentiment analysis project with a fast baseline (TF-IDF + LR) and a fine-tuned transformer (DistilBERT). Includes a Streamlit demo and Dockerfile for easy sharing.

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
