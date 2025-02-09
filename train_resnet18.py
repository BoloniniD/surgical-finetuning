import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import time
from multiprocessing import freeze_support

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_epochs = 100
batch_size = 512
learning_rate = 0.1
momentum = 0.9
weight_decay = 5e-4

# Data preprocessing and augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

def main():
    # Load CIFAR-100 dataset
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size,
                           shuffle=False, num_workers=2)

    # Initialize ResNet18 model
    model = resnet18(pretrained=False)
    # Modify the first conv layer to handle CIFAR-100's 32x32 images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool as it's not needed for small images
    # Modify final fully connected layer for 100 classes
    model.fc = nn.Linear(model.fc.in_features, 100)
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                         momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training function
    def train(epoch):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.3f}, '
                      f'Acc: {100.*correct/total:.2f}%')

    # Testing function
    def test(epoch):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100.*correct/total
        print(f'\nTest set: Average loss: {test_loss/len(testloader):.3f}, '
              f'Accuracy: {acc:.2f}%\n')
        return acc

    # Main training loop
    best_acc = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f'\nEpoch: {epoch+1}')
        train(epoch)
        acc = test(epoch)
        scheduler.step()

        # Save checkpoint
        if acc > best_acc:
            print(f'Saving checkpoint... Accuracy: {acc:.2f}%')
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            torch.save(state, 'resnet18_cifar100.pth')
            best_acc = acc

    end_time = time.time()
    print(f'Training completed in {(end_time-start_time)/60:.2f} minutes')
    print(f'Best accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    freeze_support()
    main()
