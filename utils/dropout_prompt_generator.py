import torch
import torch.nn as nn
import torch.nn.functional as F

class PromptGeneratorWithDropout(nn.Module):
    def __init__(self, embed_dim, num_prompts, dropout=0.2):
        super().__init__()
        self.prompts = nn.Parameter(torch.randn(num_prompts, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, B):
        prompts = self.prompts.unsqueeze(0).repeat(B, 1, 1)  # [B, P, D]
        return self.dropout(prompts)
