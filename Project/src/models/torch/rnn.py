from torch import nn

class SimpleRNN(nn.Module):
    def __init__(self, **config):
        """
        Initialize the SimpleRNN model.
        Args:
            input_size (int): The number of input features per timestep.
            hidden_size (int): The number of features in the hidden state.
            output_size (int): The number of output features (n_features).
        """
        super(SimpleRNN, self).__init__()
        self.lstm = nn.LSTM(**config, batch_first=True)
        self.fc = nn.Linear(in_features=config["hidden_size"], 
                            out_features=config["output_size"])

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        # output shape: (batch, seq_len, hidden_size)
        output, (h_n, c_n) = self.lstm(x)
        # # Take the last output and map to output_size
        # output[:, -1, :] gets the last timestep: (batch, hidden_size)
        out = self.fc(output[:, -1, :])  # (batch, output_size)
        # # Reshape to match target shape (batch, lookahead, n_features)
        return out.unsqueeze(1)  # (batch, 1, output_size)



def create_rnn(config):
    return SimpleLSTM(**config)
