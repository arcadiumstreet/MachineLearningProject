from torch import nn

class SimpleLSTM(nn.Module):
    def __init__(self, config):
        """
        Initialize the SimpleLSTM model.
        Args:
            config (dict): Configuration dictionary with nested structure:
                - config["lstm"]: dict with LSTM parameters (input_size, hidden_size, num_layers, dropout)
                - config["fc"]: dict with FC layer parameters (output_size)
        """
        super(SimpleLSTM, self).__init__()
        if config["lstm"]["num_layers"] == 1:
            config["lstm"]["dropout"] = 0.0
        self.lstm = nn.LSTM(**config["lstm"], batch_first=True)
        self.fc = nn.Linear(in_features=config["lstm"]["hidden_size"], 
                            out_features=config["fc"]["output_size"])

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        # output shape: (batch, seq_len, hidden_size)
        output, (h_n, c_n) = self.lstm(x)
        # Take the last output and map to output_size
        # output[:, -1, :] gets the last timestep: (batch, hidden_size)
        out = self.fc(output[:, -1, :])  # (batch, output_size)
        # Reshape to match target shape (batch, lookahead, n_features)
        return out.unsqueeze(1)  # (batch, 1, output_size)


def create_lstm(config):
    return SimpleLSTM(config)