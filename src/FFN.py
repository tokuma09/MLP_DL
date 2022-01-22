import torch.nn as nn


class TwoLayer(nn.Module):
    # TwoLayer Model for Binary classification
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.a1 = nn.ReLU()  # ReLU
        self.l2 = nn.Linear(hidden_dim, output_dim)
        self.a2 = nn.Sigmoid()

        self.layers = [self.l1, self.a1, self.l2, self.a2]

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x
