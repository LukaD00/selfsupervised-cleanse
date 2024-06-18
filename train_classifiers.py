import os
import time

import pickle

import numpy as np

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import prepare_poison_dataset

device = "cuda"
batch_size = 128

DATASET = "sig-0"  
save_name = f"{DATASET}-NEW.pt"

LOAD_CHECKPOINT = False
CHECKPOINT = ""

TRAIN = True
CLEANSE = False
CLEANSED_LABELS_NAME = f"" 

for DATASET in ["sig-1", "sig-2", 
                "badnets1-0", "badnets1-1", "badnets1-2",
                "badnets10-0", "badnets10-1", "badnets10-2"]:

    print()
    print()
    print(f"Traning {DATASET}")

    class CleansedDataset(VisionDataset):

        def __init__(self, poison_dataset: VisionDataset, cleansed_labels_name: str, transforms: torch.nn.Module = None):
            with open(f"./cleansed_labels/{cleansed_labels_name}.pkl", 'rb') as f:
                predicted_poison = pickle.load(f)

            self.data = [poison_dataset[i][0] for i in range(len(poison_dataset)) if not predicted_poison[i]]
            self.labels = [poison_dataset[i][1] for i in range(len(poison_dataset)) if not predicted_poison[i]]
            self.transforms = transforms

        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, index):
            item = self.data[index]
            label = self.labels[index]

            if self.transforms:
                item = self.transforms(item)

            return item, label

    class SkipLabelDataset(VisionDataset):

        def __init__(self, original_dataset: VisionDataset, skip_class: int):
            self.return_as_pil = type(original_dataset[0][0]) is Image.Image

            targets = np.array(original_dataset.targets)
            self.data = original_dataset.data[targets != skip_class]
            self.targets = targets[targets != skip_class].tolist()

        def __getitem__(self, index: int):
            data = self.data[index]
            target = self.targets[index]

            if self.return_as_pil:
                data = Image.fromarray(data)
            
            return data, target
        
        def __len__(self) -> int:
            return len(self.data)
        
    import random

    class ProbTransform(torch.nn.Module):
        def __init__(self, f, p=1):
            super(ProbTransform, self).__init__()
            self.f = f
            self.p = p

        def forward(self, x):
            if random.random() < self.p:
                return self.f(x)
            else:
                return x

    transform_train = transforms.Compose([
        ProbTransform(transforms.RandomCrop(32, padding=5), p=0.8),
        ProbTransform(transforms.transforms.RandomRotation(10), p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_poison_dataset, _, target_class, train_dataset = prepare_poison_dataset(DATASET, train=True, transform=transform_train, return_original_label=False)
    if CLEANSE:
        train_poison_dataset = CleansedDataset(train_poison_dataset, CLEANSED_LABELS_NAME + "-train")
    trainloader = DataLoader(train_poison_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


    test_poison_dataset, _, _, test_dataset = prepare_poison_dataset(DATASET, train=False, transform=transform_test, return_original_label=False)
    if CLEANSE:
        test_poison_dataset = CleansedDataset(test_poison_dataset, CLEANSED_LABELS_NAME + "-test")
    testloader = DataLoader(test_poison_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


    class PreActBlock(nn.Module):
        """Pre-activation version of the BasicBlock."""

        expansion = 1

        def __init__(self, in_planes, planes, stride=1):
            super(PreActBlock, self).__init__()
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.ind = None

            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
                )

        def forward(self, x):
            out = F.relu(self.bn1(x))
            shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
            out = self.conv1(out)
            out = self.conv2(F.relu(self.bn2(out)))
            if self.ind is not None:
                out += shortcut[:, self.ind, :, :]
            else:
                out += shortcut
            return out


    class PreActBottleneck(nn.Module):
        """Pre-activation version of the original Bottleneck module."""

        expansion = 4

        def __init__(self, in_planes, planes, stride=1):
            super(PreActBottleneck, self).__init__()
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
                )

        def forward(self, x):
            out = F.relu(self.bn1(x))
            shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
            out = self.conv1(out)
            out = self.conv2(F.relu(self.bn2(out)))
            out = self.conv3(F.relu(self.bn3(out)))
            out += shortcut
            return out


    class PreActResNet(nn.Module):
        def __init__(self, block, num_blocks, num_classes=10):
            super(PreActResNet, self).__init__()
            self.in_planes = 64

            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.linear = nn.Linear(512 * block.expansion, num_classes)

        def _make_layer(self, block, planes, num_blocks, stride):
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

        def forward(self, x):
            out = self.conv1(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out


    def PreActResNet18(num_classes=10):
        return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes)

    def save_model(model, optimizer, scheduler, epoch, name):
        out = os.path.join('./saved_models/preact-resnet18', name)

        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch
                    }, out)

        print(f"\tSaved model, optimizer, scheduler and epoch info to {out}")

    model = PreActResNet18()
    model.to(device)

    epochs = 35
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100,200,300,400], 0.1)

    start_epoch = 1

    if LOAD_CHECKPOINT:
        out = os.path.join('./saved_models/preact-resnet18/', CHECKPOINT)
        checkpoint = torch.load(out, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

        print("Loaded checkpoint")
        print(f"{start_epoch = }")

    best_acc = 0
    save_name = f"{DATASET}-NEW.pt"

    # Training
    def train(epoch, model, dataloader, criterion):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for _, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            if len(targets.shape)>1:
                targets = targets.squeeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        acc = 100.*correct/total
        return train_loss, acc

    def test(epoch, model, dataloader, criterion, optimizer, save=False):
        global best_acc
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)

                if len(targets.shape)>1:
                    targets = targets.squeeze(1)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            if save: save_model(model, optimizer, scheduler, epoch, save_name)
            best_acc = acc
        return test_loss, acc


    if TRAIN:
        for epoch in range(start_epoch, epochs+1):
            print(f"Epoch [{epoch}/{epochs}]\t")
            stime = time.time()

            train_loss, train_acc = train(epoch, model, trainloader, criterion)
            test_loss, test_acc = test(epoch, model, testloader, criterion, optimizer, save=True)
            scheduler.step()

            print(f"\tTraining Loss: {train_loss} Test Loss: {test_loss}")
            print(f"\tTraining Acc: {train_acc} Test Acc: {test_acc}")
            time_taken = (time.time()-stime)/60
            print(f"\tTime Taken: {time_taken} minutes")    


    if TRAIN:
        # loading the best model checkpoint from training before testing
        out = os.path.join('./saved_models/preact-resnet18/', save_name)
        checkpoint = torch.load(out, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    transform_clean = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_poison = transforms.Compose([
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    clean_test_dataset = torchvision.datasets.CIFAR10(root='C:/Datasets', train=False, download=False, transform=transform_clean)
    testloader_clean = DataLoader(clean_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    _, c_acc = test(0, model, testloader_clean, criterion, optimizer, save=False)

    test_dataset = torchvision.datasets.CIFAR10(root='C:/Datasets', train=False, download=False)
    test_dataset = SkipLabelDataset(test_dataset, target_class)
    full_poisoned_test_dataset, _, _, _ = prepare_poison_dataset(DATASET, train=False, transform=transform_poison, return_original_label=False, clean_dataset=test_dataset)

    testloader_full_poison = DataLoader(full_poisoned_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    _, asr = test(0, model, testloader_full_poison, criterion, optimizer, save=False)

    print("------")
    print(f"Clean Accuracy (C-Acc): {c_acc}")
    print(f"Attack Success Rate (ASR): {asr}")