from transformers import AutoModel, BertTokenizerFast
from deep_translator import GoogleTranslator
import torch
import numpy as np
import torch.nn as nn


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


def predict_question(input_questions):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = torch.load('models/question_type_model.pt', map_location=torch.device(device))
    bert = AutoModel.from_pretrained('bert-base-uncased')
    model = BERT_Arch(bert)
    model.load_state_dict(torch.load('models/inputs_classification_saved_weights.pt', map_location=torch.device(device)))
    max_seq_len = 30
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    input_questions = [GoogleTranslator(source='auto', target='en').translate(input_questions)]
    input_tokens = tokenizer.batch_encode_plus(
        input_questions,
        max_length=max_seq_len,
        padding=True,
        truncation=True,
        return_token_type_ids=False
        )

    input_seq = torch.tensor(input_tokens['input_ids'])
    input_mask = torch.tensor(input_tokens['attention_mask'])

    with torch.no_grad():
        predictions = model(input_seq.to(device), input_mask.to(device), return_dict=True)
        predictions = predictions.detach().cpu().numpy()

    predictions = np.argmax(predictions, axis=1)
    return predictions[0]
