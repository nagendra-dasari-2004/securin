import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('data.csv')

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Convert labels into numerical format
df['Type'] = df['Type'].astype('category').cat.codes

class ThreatDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Split dataset into training and evaluation sets
train_texts, eval_texts, train_labels, eval_labels = train_test_split(
    df['Description'].tolist(), df['Type'].tolist(), test_size=0.2)

# Tokenize training and evaluation data
train_encodings = tokenizer(train_texts, padding='max_length', truncation=True, return_tensors="pt")
eval_encodings = tokenizer(eval_texts, padding='max_length', truncation=True, return_tensors="pt")

# Create datasets
train_dataset = ThreatDataset(train_encodings, train_labels)
eval_dataset = ThreatDataset(eval_encodings, eval_labels)

# Model setup
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df['Type'].unique()))

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Fine-tune the model
trainer.train()

# Save the model
model.save_pretrained('./fine_tuned_bert')
tokenizer.save_pretrained('./fine_tuned_bert')
