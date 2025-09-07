from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate

MODEL_NAME = 'distilbert-base-uncased'

def preprocess(batch, tokenizer, max_length=256):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=max_length)

def main(output_dir='models/distilbert-imdb'):
    ds = load_dataset('imdb')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_train = ds['train'].map(lambda x: preprocess(x, tokenizer), batched=True)
    tokenized_test = ds['test'].map(lambda x: preprocess(x, tokenizer), batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    metric = evaluate.load('accuracy')
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)
    training_args = TrainingArguments(output_dir=output_dir, per_device_train_batch_size=16,
                                      per_device_eval_batch_size=32, num_train_epochs=2, eval_strategy='epoch',
                                      save_strategy='epoch', logging_steps=200, load_best_model_at_end=True,
                                      metric_for_best_model='accuracy')
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_train, eval_dataset=tokenized_test,
                      tokenizer=tokenizer, compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(output_dir)
    print('model saved to', output_dir)

if __name__ == '__main__':
    main()
