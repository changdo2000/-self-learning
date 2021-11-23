import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(28*28, 256),
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)

class unknownModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28*28, 256)
        self.l2 = nn.Linear(256, 64)
        self.l3 = nn.Linear(64, 10)
        self.do = nn.Dropout(0.1)
    
    def forward(self, x):
        k1 = nn.functional.ReLU(self.l1(x))
        k2 = nn.functional.ReLU(self.l2(k1))
        do = self.do(k2 + k1)
        y_hat = self.l3(do)
        
        return y_hat

model1 = unknownModel()
print("compile successfully")