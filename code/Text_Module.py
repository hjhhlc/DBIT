import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, bert_model, hidden_dim, n_layers):
        super(TextEncoder, self).__init__()
        self.bert = bert_model
        self.lstm = nn.LSTM(input_size=bert_model.config.hidden_size,
                            hidden_size=hidden_dim,
                            num_layers=n_layers,
                            batch_first=True,
                            bidirectional=True)
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        _, h_n = self.lstm(sequence_output)
        h = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # 双向
        return self.linear(h)

