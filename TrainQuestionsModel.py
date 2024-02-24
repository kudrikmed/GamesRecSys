import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModel, BertTokenizerFast, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import mlflow

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Load and preprocess data
def load_and_preprocess_data(file_path, label):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        df = pd.DataFrame({'question': lines, 'gaming': label})
    return df


relevant_questions_path = 'data/raw/gaming_questions.txt'
irrelevant_questions_path = 'data/raw/non_gaming_questions.txt'
df_gaming = load_and_preprocess_data(relevant_questions_path, 1)
df_non_gaming = load_and_preprocess_data(irrelevant_questions_path, 0)

df = pd.concat([df_gaming, df_non_gaming], ignore_index=True).sample(frac=1).reset_index(drop=True)

# Train-test-validation split
train_text, temp_text, train_labels, temp_labels = train_test_split(df['question'], df['gaming'],
                                                                    random_state=42, test_size=0.3,
                                                                    stratify=df['gaming'])
val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                random_state=42, test_size=0.5,
                                                                stratify=temp_labels)

# BERT initialization and tokenization
bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
max_seq_len = 30


def tokenize_sequences(text):
    return tokenizer.batch_encode_plus(
        text.tolist(),
        max_length=max_seq_len,
        padding=True,
        truncation=True,
        return_token_type_ids=False
    )


tokens_train = tokenize_sequences(train_text)
tokens_val = tokenize_sequences(val_text)
tokens_test = tokenize_sequences(test_text)


# Convert to PyTorch tensors
def convert_to_tensors(tokens):
    return torch.tensor(tokens['input_ids']), torch.tensor(tokens['attention_mask'])


train_seq, train_mask = convert_to_tensors(tokens_train)
val_seq, val_mask = convert_to_tensors(tokens_val)
test_seq, test_mask = convert_to_tensors(tokens_test)

# Define batch size
batch_size = 32

# Wrap tensors in TensorDataset
train_data = TensorDataset(train_seq, train_mask, torch.tensor(train_labels.tolist()))
val_data = TensorDataset(val_seq, val_mask, torch.tensor(val_labels.tolist()))

# Create data loaders
train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)
val_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=batch_size)

# Freeze BERT parameters
for param in bert.parameters():
    param.requires_grad = False


# BERT Architecture
class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask, return_dict):
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


# Initialize model, optimizer, and loss function
model = BERT_Arch(bert).to(device)
optimizer = AdamW(model.parameters(), lr=1e-3)
class_wts = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
weights = torch.tensor(class_wts, dtype=torch.float).to(device)
cross_entropy = nn.NLLLoss(weight=weights)


# Training and evaluation functions
def train():
    model.train()
    total_loss, total_preds = 0, []

    for step, batch in enumerate(train_dataloader):
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        model.zero_grad()
        preds = model(sent_id, mask, return_dict=False)
        loss = cross_entropy(preds, labels)
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        preds = preds.detach().cpu().numpy()
        total_preds.append(preds)

    avg_loss = total_loss / len(train_dataloader)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds


def evaluate():
    model.eval()
    total_loss, total_preds = 0, []

    for step, batch in enumerate(val_dataloader):
        batch = [t.to(device) for t in batch]
        sent_id, mask, labels = batch

        with torch.no_grad():
            preds = model(sent_id, mask, return_dict=False)
            loss = cross_entropy(preds, labels)
            total_loss += loss.item()
            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)

    avg_loss = total_loss / len(val_dataloader)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds


# Training loop with MLflow logging
with mlflow.start_run():
    mlflow.log_params({
        'model_name': 'bert-base-uncased',
        'gaming_questions_path': relevant_questions_path,
        'nongaming_questions_path': irrelevant_questions_path
    })

    epochs = 20
    best_valid_loss = float('inf')
    train_losses = []
    valid_losses = []

    # Log parameters
    mlflow.log_param('epochs', epochs)
    mlflow.log_param('learning_rate', 1e-3)
    mlflow.log_param('batch_size', batch_size)

    for epoch in range(epochs):
        print(f'\n Epoch {epoch + 1} / {epochs}')
        train_loss, _ = train()
        valid_loss, _ = evaluate()

        # Log metrics
        mlflow.log_metric('train_loss', train_loss, step=epoch)
        mlflow.log_metric('valid_loss', valid_loss, step=epoch)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'models/inputs_classification_saved_weights.pt')
            mlflow.pytorch.log_model(model, 'models')

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')

# Load the best model and evaluate on the test set
model.load_state_dict(torch.load('models/inputs_classification_saved_weights.pt'))
with torch.no_grad():
    preds = model(test_seq.to(device), test_mask.to(device), return_dict=True)
    preds = preds.detach().cpu().numpy()

preds = np.argmax(preds, axis=1)
print(classification_report(test_labels, preds))
