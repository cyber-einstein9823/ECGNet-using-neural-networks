import math
import torch
import torch.nn as nn
class ECGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(128, 187) * math.sqrt(2.0 / 187))
        self.b1 = nn.Parameter(torch.zeros(128))
        self.W2 = nn.Parameter(torch.randn(64, 128) * math.sqrt(2.0 / 128))
        self.b2 = nn.Parameter(torch.zeros(64))
        self.W3 = nn.Parameter(torch.randn(5, 64) * math.sqrt(2.0 / 64))
        self.b3 = nn.Parameter(torch.zeros(5))
        self.last_A2 = None
    def forward(self, x):
        a1 = torch.clamp(x @ self.W1.T + self.b1, min=0)
        a2 = torch.clamp(a1 @ self.W2.T + self.b2, min=0)
        self.last_A2 = a2
        return a2 @ self.W3.T + self.b3
def cross_entropy(logits, y):
    lse = torch.logsumexp(logits, dim=1)
    return (-logits[torch.arange(logits.shape[0], device=logits.device), y] + lse).mean()
