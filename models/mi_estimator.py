import torch
import torch.nn as nn

class MIEstimator(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x1, x2):
        # x1, x2: [N, D] (N có thể là B hoặc C)
        if x1.dim() != 2 or x2.dim() != 2:
            x1 = x1.view(x1.size(0), -1)
            x2 = x2.view(x2.size(0), -1)

        if x1.size(0) != x2.size(0):
            raise ValueError(f"MIEstimator: batch mismatch {x1.size(0)} vs {x2.size(0)}")

        combined = torch.cat([x1, x2], dim=-1)  # <- dim=-1 an toàn hơn dim=1
        return self.mlp(combined).squeeze(-1)   # -> [N]