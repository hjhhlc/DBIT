import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, bert_model, hidden_dim):
        super(TextEncoder, self).__init__()
        self.bert = bert_model
        self.gru = nn.GRU(input_size=bert_model.config.hidden_size,
                          hidden_size=hidden_dim,
                          batch_first=True,
                          bidirectional=True)
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        _, h_n = self.gru(sequence_output)
        h = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # 双向
        return self.linear(h)
