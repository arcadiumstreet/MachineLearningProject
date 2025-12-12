import torch
import torch.nn as nn

class TextAttentionNet(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=64):
        super(TextAttentionNet, self).__init__()
        # 1. Per-Headline Processor
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, 256),
            # nn.BatchNorm1d(256), # Added Batch Normalization
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, hidden_dim),
            # nn.BatchNorm1d(hidden_dim), # Added Batch Normalization
            nn.Tanh()
        )

        # 2. Attention Aggregator (Many headlines -> One vector)
        # Replaced learnable attention with fixed positional weighting

        # 3. Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32), # Added Batch Normalization
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (Batch, Max_Headlines, Embed_Dim)

        # Pass features: (Batch, Max_H, Hidden)
        # Note: BatchNorm1d expects input of shape (N, C, L) or (N, C)
        # Our input x for encoder is (Batch, Max_H, Embed_Dim).
        # To apply BatchNorm1d correctly, we need to apply it per feature set
        # or reshape if applying to the whole sequence. Given the current architecture,
        # where encoder processes 'per-headline', it implies each 'headline' (embedding)
        # is processed independently, so BatchNorm will apply to the feature dimension.
        # The current implementation of encoder is good, BatchNorm1d will apply to the last dimension.
        features = self.encoder(x)

        # Calculate Fixed Positional Weights based on exp(-x/7)
        max_headlines = features.shape[1]
        indices = torch.arange(max_headlines, device=features.device, dtype=torch.float32)
        raw_positional_weights = torch.exp(-indices / 7.0)
        # Normalize weights so they sum to 1
        normalized_positional_weights = raw_positional_weights / raw_positional_weights.sum()

        # Reshape for broadcasting across batch and hidden dimensions
        # attn_weights_for_sum shape: (1, Max_Headlines, 1)
        attn_weights_for_sum = normalized_positional_weights.unsqueeze(0).unsqueeze(-1)

        # Weighted Sum: (Batch, Hidden)
        context_vector = torch.sum(features * attn_weights_for_sum, dim=1)

        # Final Probability
        return self.classifier(context_vector)