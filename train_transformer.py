import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

DATA_FILE = 'SIRE.xlsx'
TEXT_COL = 'Descripción detallada del evento'
LABEL_COL = 'Tipo evento'
MODEL_NAME = 'dccuchile/bert-base-spanish-wwm-uncased'

# Load Excel file
print('Loading data...')
df = pd.read_excel(DATA_FILE)

# Drop rows with missing values in key columns
clean_df = df[[TEXT_COL, LABEL_COL]].dropna()

# Encode labels as categorical ids
labels = sorted(clean_df[LABEL_COL].unique())
label2id = {label: i for i, label in enumerate(labels)}
clean_df['label'] = clean_df[LABEL_COL].map(label2id)

# Split data
train_df, eval_df = train_test_split(clean_df, test_size=0.2, random_state=42, stratify=clean_df['label'])

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Dataset class
class IncidentDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.texts = list(dataframe[TEXT_COL])
        self.labels = list(dataframe['label'])
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=128)
        enc['labels'] = self.labels[idx]
        return {k: torch.tensor(v) for k, v in enc.items()}

# Create datasets
ds_train = IncidentDataset(train_df)
ds_eval = IncidentDataset(eval_df)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(labels))

args = TrainingArguments(
    output_dir='results',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy='epoch',
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_train,
    eval_dataset=ds_eval,
)

trainer.train()
