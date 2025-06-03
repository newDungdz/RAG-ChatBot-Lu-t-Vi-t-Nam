# Import necessary libraries
# import fasttext
import numpy as np
import json
import unicodedata
from collections import Counter
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from datasets import Dataset

def read_json_file(json_file_path: str):
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def save_to_json(data, output_path):
    # Save as JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved: {output_path}")
# Step 1: Data Preparation


def load_data(json_filepath: str) -> pd.DataFrame:
    """
    Load and preprocess data from a JSON file.
    Assumes JSON structure: {category: [text1, text2, ...], ...}
    Treats each text as potentially having multiple labels.
    """
    raw_data = read_json_file(json_filepath)
    
    # Convert to DataFrame
    df = pd.DataFrame(raw_data, columns=["label", "text"])
    return df

# Prepare data
filepath = 'questions.json'  # Update with actual path
df = load_data(filepath)

print(df)

multi_label_rows = df[df["label"].apply(len) > 1]
print(f"Number of multi-label rows: {len(multi_label_rows)}")

# Binarize labels
mlb = MultiLabelBinarizer()
df['label'] = mlb.fit_transform(df['label']).tolist()

print("GPU available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Split into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Load BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(mlb.classes_))

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./bert_multi_label')
tokenizer.save_pretrained('./bert_multi_label')

print("Training complete and model saved.")
