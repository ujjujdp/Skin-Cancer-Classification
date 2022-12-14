### Imports
import torch
import torch.nn as nn
import torch.optim as optim

from data import loaders
from model import Classifier_net
from check import train, test

### Tunable Hyperparameters (to change fully connected layer, modify model.py)
lr = 0.0001
epochs = 40

### Optimizer and Loss definition
model_criterion = nn.CrossEntropyLoss()
model_optimizer = optim.Adam(Classifier_net.parameters(), lr)

### checking for hardware accelarators
accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'
# accelerator = 'cpu'
print("Device =", accelerator)

### begin training loop
Classifier_net = train(epochs, loaders, Classifier_net, model_optimizer, model_criterion, accelerator, './Classifier_net.pt', verbose=True)

### begin test loop
test(loaders, Classifier_net, model_criterion, accelerator, verbose=True)