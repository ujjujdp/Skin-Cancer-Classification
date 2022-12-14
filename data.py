### Imports
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# training set
train_transforms = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])

# test and valid sets
check_transforms = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])

### Loading files as a dataset
trainset = datasets.ImageFolder('./data/train/', transform=train_transforms)
testset = datasets.ImageFolder('./data/testData/', transform=check_transforms)
validset = datasets.ImageFolder('./data/valid/', transform=check_transforms)

### Creating batches of iterables from the datasets
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=True)
validloader = DataLoader(validset, batch_size=64, shuffle=True)

loaders = {'train':trainloader, 'test':testloader, 'valid':validloader}