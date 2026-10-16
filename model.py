# model.py
import torch
import torch.nn as nn


class HandTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_dim: int = 63,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
        max_len: int = 30,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.input_dim = input_dim
        self.d_model = d_model
        self.max_len = max_len

        self.input_proj = nn.Linear(input_dim, d_model)

        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_len, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, 21, 3]
        return: [B, num_classes]
        """

        if x.dim() != 4:
            raise ValueError(f"Expected x shape [B,T,21,3], got {x.shape}")

        B, T, J, C = x.shape

        if J != 21 or C != 3:
            raise ValueError(f"Expected landmarks shape [21,3], got [{J},{C}]")

        x = x.reshape(B, T, J * C)      # [B, T, 63]
        x = self.input_proj(x)          # [B, T, d_model]
        x = x + self.pos_embedding[:, :T, :]

        out = self.transformer(x)       # [B, T, d_model]

        feat = out.mean(dim=1)          # 平均 30 幀特徵
        logits = self.classifier(feat)

        return logits


