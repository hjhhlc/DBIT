import torch.nn as nn
import torch.nn.functional as F


class PinyinEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PinyinEncoder, self).__init__()
        self.depthwise = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1, groups=input_dim)
        self.pointwise = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):  # x: [B, seq_len, input_dim]
        x = x.permute(0, 2, 1)  # -> [B, input_dim, seq_len]
        x = self.depthwise(x)
        x = F.relu(self.pointwise(x))  # -> [B, hidden_dim, seq_len]
        x = self.pool(x).squeeze(-1)   # -> [B, hidden_dim]
        return x
