import torch.nn as nn
import torch.nn.functional as F


class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, hidden_size=128, out_dim=64,
                 dropout=0.1, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_size, num_layers=1, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_dim),
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(self.embedding(x))
        z = self.proj(hn.squeeze(0))
        return F.normalize(z, dim=1) if self.normalize else z
