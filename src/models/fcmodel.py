import torch
import torch.nn as nn


class FCModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, name=None):
        super(FCModel, self).__init__()
        self.model_name = name if name else self.__class__.__name__
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        return x


data = torch.randn((1, 1, 10000))
model = FCModel(input_size=10000, hidden_size=1024, output_size=784)
out = model(data)
print("input Shape:", data.shape)
print("out shape:", out.shape)
