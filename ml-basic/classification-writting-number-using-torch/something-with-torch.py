import torch
import torch.nn as nn
from torch import optim

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
loss  = nn.CrossEntropyLoss()

params = model.parameters()
optimizer = optim.SGD(params, lr = 1e-2)

from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

train_data = datasets.MNIST('data', train = True, download = True, transform = transforms.ToTensor())

train, val = random_split(train_data, [55000, 5000])

train_loader = DataLoader(train, batch_size = 32)
val_loader = DataLoader(val, batch_size = 32)

number_epochs = 20

for epoch in range(number_epochs):
    losses = list()
    for batch in train_loader:
        x, y = batch
        b = x.size(0)
        x = x.view(b, -1)

        y_hat = model(x)

        J = loss(y_hat, y)

        model.zero_grad()

        J.backward()

        optimizer.step()
        losses.append(J.item)
    print("Eps: ", epoch + 1, 'Train loss', torch.tensor(J).mean())
print("compile successfully")