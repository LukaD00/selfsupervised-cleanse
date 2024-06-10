import random

import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.datasets.vision import VisionDataset

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from typing import Tuple


class SIGTriggerHandler():

    def __init__(self, delta, freq, alpha=0.2):
        super(SIGTriggerHandler, self).__init__()

        self.delta = delta
        self.freq = freq
        self.alpha = alpha

        img_shape = (32, 32, 3)
        self.pattern = np.zeros(img_shape, dtype=np.float32)
        m = self.pattern.shape[1]
        for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                for k in range(img_shape[2]):
                    self.pattern[i, j] = delta * np.sin(2 * np.pi * j * self.freq / m)

    def put_trigger(self, img):
        #img = functional.pil_to_tensor(img)
        img = np.array(img)
        img = np.float32(img)
        img = self.alpha * np.uint32(img) + (1 - self.alpha) * self.pattern
        img = np.uint8(np.clip(img, 0, 255))
        img = Image.fromarray(img)
        return img


class SIGDataset(VisionDataset):
    
    def __init__(self, original_dataset: VisionDataset, target_class: int, delta: float, freq: float, alpha: float = 0.2,
                 transform: torch.nn.Module = None, seed: int = None, poisoning_rate: float = 0.1, 
                 return_original_label: bool = True, attack_test: bool = False):
        
        self.to_tensor = ToTensor()

        self.original_dataset = original_dataset
        self.transform = transform
        self.return_original_label = return_original_label
        self.target_class = target_class
        self.poisoning_rate = poisoning_rate
        self.attack_test = attack_test
        self.trigger_handler = SIGTriggerHandler(delta, freq, alpha)

        indices = [i for i in range(len(original_dataset.targets)) if original_dataset.targets[i]==target_class]
        if seed: random.seed(seed)
        number_poisoned = min(int(len(indices) * self.poisoning_rate), len(indices))
        self.poi_indices = random.sample(indices, k=number_poisoned)

    def __getitem__(self, index):
        img, original_label = self.original_dataset[index]

        # NOTE: According to the threat model, the trigger should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices or self.attack_test:
            poison_label = self.target_class
            img = self.trigger_handler.put_trigger(img)
        else:
            poison_label = original_label

        # WaNet attack needs to transform images to Pytorch tensors, so we by default transform into tensors
        # here as well to make it consistent
        if not torch.is_tensor(img):
            img = self.to_tensor(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.return_original_label:
            return img, poison_label, original_label
        else:
            return img, poison_label
    
    def __len__(self):
        return len(self.original_dataset)
    
    def is_poison(self, index):
        return index in self.poi_indices


class BadNetsTriggerHandler():

    def __init__(self, trigger_path, trigger_size, img_width, img_height):
        self.trigger_img = Image.open(trigger_path).convert('RGB')
        self.trigger_size = trigger_size
        self.trigger_img = self.trigger_img.resize((trigger_size, trigger_size))        
        self.img_width = img_width
        self.img_height = img_height

    def put_trigger(self, img):
        img.paste(self.trigger_img, (self.img_width - self.trigger_size, self.img_height - self.trigger_size))
        return img


class BadNetsDataset(VisionDataset):

    def __init__(self, original_dataset: VisionDataset, target_class: int, trigger_path: str, 
                 transform: torch.nn.Module = None, trigger_size: int = 3, seed: int = None, poisoning_rate: float = 0.1,
                 return_original_label: bool = True):

        self.to_tensor = ToTensor()

        self.original_dataset = original_dataset
        self.transform = transform
        self.return_original_label = return_original_label
        self.target_class = target_class
        self.poisoning_rate = poisoning_rate
        
        self.width, self.height, self.channels = original_dataset.data.shape[1:]
        self.trigger_handler = BadNetsTriggerHandler(trigger_path, trigger_size, self.width, self.height)

        indices = [i for i in range(len(original_dataset.targets)) if original_dataset.targets[i]!=target_class]
        if seed: random.seed(seed)
        number_poisoned = min(int(len(original_dataset.targets) * self.poisoning_rate), len(indices))
        self.poi_indices = random.sample(indices, k=number_poisoned)

    def __getitem__(self, index):
        img, original_label = self.original_dataset[index]

        # NOTE: According to the threat model, the trigger should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices:
            poison_label = self.target_class
            img = self.trigger_handler.put_trigger(img)
        else:
            poison_label = original_label

        # WaNet attack needs to transform images to Pytorch tensors, so we by default transform into tensors
        # here as well to make it consistent
        if not torch.is_tensor(img):
            img = self.to_tensor(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.return_original_label:
            return img, poison_label, original_label
        else:
            return img, poison_label
    
    def __len__(self):
        return len(self.original_dataset)

    def is_poison(self, index):
        return index in self.poi_indices

class WaNetDataset(VisionDataset):

    def __init__(self, 
                 original_dataset: VisionDataset, 
                 target_class: int, 
                 transform: torch.nn.Module = None, 
                 seed: int = None, 
                 poisoning_rate: float = 0.1, 
                 noise_rate: float = 0.2,
                 k: float = 4, 
                 s: float = 0.5, 
                 device: str = "cpu", 
                 return_original_label: bool = True):

        self.to_tensor = ToTensor()

        self.original_dataset = original_dataset
        self.transform = transform
        self.return_original_label = return_original_label

        self.target_class = target_class
        
        self.width, self.height, self.channels = original_dataset.data.shape[1:]
        self.device = device

        indices = [i for i in range(len(original_dataset.targets)) if original_dataset.targets[i]!=target_class]
        if seed: 
            random.seed(seed)
            torch.manual_seed(seed)
        number_poisoned = min(int(len(original_dataset.targets) * poisoning_rate), len(indices))
        self.poi_indices = random.sample(indices, k=number_poisoned)

        clean_indices = [i for i in range(len(original_dataset.targets)) if i not in self.poi_indices]
        self.noise_indices = random.sample(clean_indices, k=int(len(original_dataset.targets) * noise_rate))

        # Prepare attack grid
        ins = torch.rand(1, 2, k, k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        attack_grid = (
            F.interpolate(ins, size=self.height, mode="bicubic", align_corners=True)
            .permute(0, 2, 3, 1)
            .to(device)
        )
        array1d = torch.linspace(-1, 1, steps=self.height)
        x, y = torch.meshgrid(array1d, array1d, indexing="ij")
        identity_grid = torch.stack((y, x), 2)[None, ...].to(device)

        attack_grid = (identity_grid + s * attack_grid / self.height) #* opt.grid_rescale
        attack_grid = torch.clamp(attack_grid, -1, 1)
        attack_grid = attack_grid.to(device)
        self.attack_grid = attack_grid


    def __getitem__(self, index):
        img, original_label = self.original_dataset[index]

        if not torch.is_tensor(img):
            img = self.to_tensor(img)
        
        if index in self.poi_indices:
            poison_label = self.target_class

            img = img.unsqueeze(0)
            img = img.type(torch.FloatTensor)
            img = img.to(self.device)
            img = F.grid_sample(img, self.attack_grid, align_corners=True)
            img = img.squeeze(0)
            img[img>255] = 255
            img[img<0] = 0

        elif index in self.noise_indices:
            poison_label = original_label
            
            ins = torch.rand(1, self.height, self.height, 2).to(self.device) * 2 - 1
            noise_grid = self.attack_grid + ins / self.height
            noise_grid = torch.clamp(noise_grid, -1, 1)

            img = img.unsqueeze(0)
            img = img.type(torch.FloatTensor)
            img = img.to(self.device)
            img = F.grid_sample(img, noise_grid, align_corners=True)
            img = img.squeeze(0)
            img[img>255] = 255
            img[img<0] = 0
        else:
            poison_label = original_label

        if self.transform is not None:
            img = self.transform(img)

        if self.return_original_label:
            return img, poison_label, original_label
        else:
            return img, poison_label

    def __len__(self):
        return len(self.original_dataset)

    def is_poison(self, index):
        return index in self.poi_indices


def show(dataset, index):
    img = dataset[index][0]

    if not torch.is_tensor(img):
        img = ToTensor()(img)

    img = img.permute(1, 2, 0)
    
    print(img)
    plt.axis('off')
    plt.imshow(img)
    plt.show()

def save(dataset, index, output_name):
    img = dataset[index][0]

    if not torch.is_tensor(img):
        img = ToTensor()(img)

    img = img.permute(1, 2, 0)
    
    plt.axis('off')
    plt.imshow(img)
    plt.savefig(output_name, bbox_inches='tight', pad_inches = 0)

def save_difference(dataset1, dataset2, index, output_name):
    img1 = dataset1[index][0]
    if not torch.is_tensor(img1):
        img1 = ToTensor()(img1)
    img1 = img1.permute(1, 2, 0)

    img2 = dataset2[index][0]
    if not torch.is_tensor(img2):
        img2 = ToTensor()(img2)
    img2 = img2.permute(1, 2, 0)

    img = img2 - img1
    
    plt.axis('off')
    plt.imshow(img)
    plt.savefig(output_name, bbox_inches='tight', pad_inches = 0)


def prepare_poison_dataset(
        dataset_name: str, 
        train: bool, 
        dataset_root: str = 'C:/Datasets', 
        download: bool = False,
        transform: torch.nn.Module = None,
        return_original_label: bool = True,
        clean_dataset: VisionDataset = None) -> Tuple[VisionDataset, np.array, int, VisionDataset]:
    
    if clean_dataset is None:
        clean_dataset = torchvision.datasets.CIFAR10(root=dataset_root, train=train, download=download)

    if dataset_name == "badnets0":
        target_class = 0
        poison_dataset = BadNetsDataset(clean_dataset, target_class, "triggers/trigger_10.png", seed=1, transform=transform, return_original_label=return_original_label)
    elif dataset_name == "badnets1":
        target_class = 1
        poison_dataset = BadNetsDataset(clean_dataset, target_class, "triggers/trigger_10.png", seed=1, transform=transform, return_original_label=return_original_label)
    elif dataset_name == "badnets2":
        target_class = 2
        poison_dataset = BadNetsDataset(clean_dataset, target_class, "triggers/trigger_10.png", seed=1, transform=transform, return_original_label=return_original_label)
    elif dataset_name == "wanet0":
        target_class = 0
        poison_dataset = WaNetDataset(clean_dataset, target_class, seed=1, transform=transform, return_original_label=return_original_label)
    elif dataset_name == "wanet1":
        target_class = 1
        poison_dataset = WaNetDataset(clean_dataset, target_class, seed=1, transform=transform, return_original_label=return_original_label)
    elif dataset_name == "wanet2":
        target_class = 2
        poison_dataset = WaNetDataset(clean_dataset, target_class, seed=1, transform=transform, return_original_label=return_original_label)
    elif dataset_name == "sig0":
        target_class = 0
        poison_dataset = SIGDataset(clean_dataset, target_class, 20, 6, seed=1, transform=transform, return_original_label=return_original_label)
    elif dataset_name == "sig1":
        target_class = 1
        poison_dataset = SIGDataset(clean_dataset, target_class, 20, 6, seed=1, transform=transform, return_original_label=return_original_label)
    elif dataset_name == "sig2":
        target_class = 2
        poison_dataset = SIGDataset(clean_dataset, target_class, 20, 6, seed=2, transform=transform, return_original_label=return_original_label)
    elif dataset_name == "clean":
        target_class = None
        transform_clean = transforms.Compose([transforms.ToTensor(), transform])
        poison_dataset = torchvision.datasets.CIFAR10(root=dataset_root, train=train, download=download, transform=transform_clean)
    else:
        #
        raise Exception("Invalid dataset")

    if dataset_name == "clean":
        poison_indices = np.zeros(len(poison_dataset))
    else:
        poison_indices = np.array([poison_dataset.is_poison(i) for i in range(len(poison_dataset))])
    

    return poison_dataset, poison_indices, target_class, clean_dataset


if __name__=="__main__":
    dataset = torchvision.datasets.CIFAR10(root='C:/Datasets', train=True, download=True)
    #poison_dataset = WaNetDataset(dataset, 0, seed=1)
    #poison_dataset = BadNetsDataset(dataset, 1, "triggers/trigger_10.png", seed=1)
    poison_dataset = SIGDataset(dataset, target_class=1, delta=20, freq=6, seed=1)

    poison_index = poison_dataset.poi_indices[3]

    #show(dataset, poison_index)
    #show(poison_dataset, poison_index)

    save(dataset, poison_index, "images/sig_no_trigger_image.png")
    save(poison_dataset, poison_index, "images/sig_trigger_image.png")
    save_difference(dataset, poison_dataset, poison_index, "images/sig_difference_image.png")