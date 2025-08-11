import torch
import torchvision
import torchvision.datasets as datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)

X = torch.from_numpy(train_set.data.transpose(0, 3, 1, 2)) / 255  # Shape: [50000, 3, 32, 32]
X = X.to(device)
y = torch.tensor(train_set.targets).to(device)  # Shape: (50000,)

from hardcoded_transforms import transforms
val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms('cifar10_T'))
test_loader =  torch.utils.data.DataLoader(val_set, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

from steps import feature_extraction_step
from torchiteration import predict, predict_classification_step


import bayeslap

be_calculator = lambda x, y, num=10, sigma=0.3, batch=32: bayeslap.BayesErrorRBF.apply(x, y, sigma, num, batch)

def train(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    acc = 100. * correct / total
    return running_loss / len(loader), acc

# Evaluation function
def evaluate(model, loader, criterion):
    model.eval()
    loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    acc = 100. * correct / total
    return loss / len(loader), acc



num_epochs = 30
batch_size = 128 
learning_rate = 0.1
weight_decay = 5e-4
momentum = 0.9


import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR



from torch.utils.data import TensorDataset

class TransformTensorDataset(TensorDataset):
    def __init__(self, *tensors, transform=None):
        super().__init__(*tensors)
        self.transform = transform

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        if self.transform:
            x = self.transform(x)
        return x, y

from torchadversarial import Attack
from tqdm import tqdm

for x_ in Attack(torch.optim.AdamW, [X], steps = 2, foreach=False, maximize=True):

    with torch.no_grad():
        x_[0].copy_(x_[0].clamp(X - 8/255, X + 8/255).clamp(0, 1))  # ← modifies in-place, but autograd is off

    model = torch.hub.load(
        'cat-claws/nn',
        'resnet_cifar',
        block='',
        layers=[2, 2, 2, 2],
        num_classes=10,
    ) # ResNet-18 from He et al. 2016

    model = model.to(device).train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                        momentum=momentum, weight_decay=weight_decay)

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    # num_epochs += 1

    p_train_loader =  torch.utils.data.DataLoader(TransformTensorDataset(x_[0].detach().clone(), y, transform=None), batch_size=batch_size, shuffle=False)

    # Training loop
    best_acc = 0
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, p_train_loader, optimizer, criterion)
        # test_loss, test_acc = evaluate(model, test_loader, criterion)
        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Acc: {train_acc:.2f}% ")

    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"Epoch [{epoch+1}/{num_epochs}] "
        f"Test Acc: {test_acc:.2f}% ")

    model.fc = torch.nn.Identity()
    model = model.to(device).eval()
    
    # normalized_images = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(x_[0])
    error = bayeslap.BayesErrorImageFeature.apply(x_[0], y, model, be_calculator)
    print(f"{error.item():.3f}")

    error.backward()


    p_data = x_[0].detach().clamp(X - 8/255, X + 8/255).clamp(0, 1)


    (p_data - X).max()

    p_loader =  torch.utils.data.DataLoader(torch.utils.data.TensorDataset(p_data, y), batch_size=batch_size, shuffle=False)
    outputs = predict(model, feature_extraction_step, val_loader = p_loader, device=device)
    p_features = torch.tensor(outputs['predictions'])
