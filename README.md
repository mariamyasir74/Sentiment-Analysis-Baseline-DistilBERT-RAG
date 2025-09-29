# Sentiment-Analysis-DistilBERT-Baseline
An end-to-end sentiment analysis project with a fast baseline (TF-IDF + LR) and a fine-tuned transformer (DistilBERT). Includes a Streamlit demo and Dockerfile for easy sharing.
## Quickstart
1. Clone repo
2.
```bash
pip install -r requirements.txt
```
4. Train baseline:
```bash
python src/baseline.py
```

<img width="1090" height="526" alt="baseline" src="https://github.com/user-attachments/assets/c8cec855-5785-4866-b148-6549f8d672cf" />

6. Train transformer:
```bash
python src/transformers_train.py
```

<img width="1615" height="150" alt="transformers_train" src="https://github.com/user-attachments/assets/222ae642-b3ce-4a47-81b8-31d4ad51297b" />

8. Run demo:
```bash
streamlit run app/streamlit_app.py
```

<img width="936" height="509" alt="Sentiment Analyzer - Demo" src="https://github.com/user-attachments/assets/9fd16c9b-2d2a-4d94-b54c-0c7e8d1a66f0" />

## Results

- Baseline accuracy:

<img width="729" height="797" alt="Baseline_negative_ex" src="https://github.com/user-attachments/assets/35716b9d-e68f-4443-aa22-569b8e6110d9" />
 
<img width="734" height="802" alt="Baseline_positive_ex" src="https://github.com/user-attachments/assets/0cc46077-70a8-4b4d-8374-c0073b537b1b" />
 
- DistilBERT accuracy:

<img width="453" height="746" alt="Transformer_negative_ex1" src="https://github.com/user-attachments/assets/0bb4780b-2343-4262-967a-51efc39a4171" />

<img width="457" height="122" alt="Transformer_negative_ex2" src="https://github.com/user-attachments/assets/f1d14d21-520d-41cc-b49f-254193507a31" />

<img width="456" height="748" alt="Transformer_positive_ex1" src="https://github.com/user-attachments/assets/b18770a8-5ccf-459e-857b-43c5606aa0e5" />

<img width="459" height="127" alt="Transformer_positive_ex2" src="https://github.com/user-attachments/assets/0f2c718d-3d46-45d9-9ad8-9f68883844ff" />
