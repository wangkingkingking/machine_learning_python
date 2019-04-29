import torch
import torch.nn as nn


class LinearSVM(nn.Module):
    
    def __init__(self):
        super(LinearSVM, self).__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)
