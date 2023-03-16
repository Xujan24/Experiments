from torch import nn, Tensor, cat
import torch

class Decoder (nn.Module):
    def __init__(self, n_input: int, n_output: int, p: float = 0.25) -> None:
        super().__init__()
        self.seq_layers = nn.Sequential(
            nn.Linear(n_input * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(1024, n_output),
            nn.Softmax(dim=1)
        )

    def forward(self, input, icd_version):
        input = cat((input, icd_version), 1)
        return self.seq_layers(input)
