### Imports
from torchvision import models
import torch.nn as nn

### Importing the model (inception v3 CNN) pretrained on Imagenet
Classifier_net = models.inception_v3()

# Note: Modify final fully connected network by using nn.Sequential() to define layers

Classifier_net.fc = nn.Sequential(nn.Linear(2048, 512),
				nn.ReLU(),
				nn.Dropout(p=0.25),
				nn.Linear(512, 64),
				nn.ReLU(),
				nn.Dropout(p=0.25),
				nn.Linear(64, 3))