import torch
import torch.nn as nn
import torch.nn.functional as F


class DBITModel(nn.Module):
    def __init__(self, text_encoder, pinyin_encoder, hidden_dim, num_classes, temperature=0.07):
        super(DBITModel, self).__init__()
        self.text_encoder = text_encoder  # e.g., TinyBERT + Grouped LSTM
        self.pinyin_encoder = pinyin_encoder  # e.g., DW-CNN
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.temperature = temperature

    def forward(self, input_ids, attention_mask, pinyin_inputs, labels=None):
        # 分别提取文本和拼音特征
        text_feat = self.text_encoder(input_ids, attention_mask)  # shape: (B, H)
        pinyin_feat = self.pinyin_encoder(pinyin_inputs)  # shape: (B, H)

        # 拼接后做分类
        fusion_feat = torch.cat([text_feat, pinyin_feat], dim=1)
        logits = self.fc(fusion_feat)

        # 对比损失
        contrastive_loss = self.contrastive(text_feat, pinyin_feat)

        # 分类损失
        if labels is not None:
            ce_loss = F.cross_entropy(logits, labels)
            total_loss = ce_loss + contrastive_loss
            return logits, total_loss
        else:
            return logits

    def contrastive(self, text_feat, pinyin_feat):
        # normalize
        text_norm = F.normalize(text_feat, dim=1)
        pinyin_norm = F.normalize(pinyin_feat, dim=1)

        # cosine similarity matrix
        similarity_matrix = torch.matmul(text_norm, pinyin_norm.T)  # (B, B)
        logits = similarity_matrix / self.temperature

        labels = torch.arange(text_feat.size(0)).to(text_feat.device)
        loss1 = F.cross_entropy(logits, labels)  # text -> pinyin
        loss2 = F.cross_entropy(logits.T, labels)  # pinyin -> text

        return (loss1 + loss2) / 2
