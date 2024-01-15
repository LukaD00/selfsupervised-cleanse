import random

from matplotlib import pyplot as plt

import torch
from torchvision.datasets.vision import VisionDataset

from PIL import Image


class TriggerHandler(object):

    def __init__(self, trigger_path, trigger_size, img_width, img_height):
        self.trigger_img = Image.open(trigger_path).convert('RGB')
        self.trigger_size = trigger_size
        self.trigger_img = self.trigger_img.resize((trigger_size, trigger_size))        
        self.img_width = img_width
        self.img_height = img_height

    def put_trigger(self, img):
        img.paste(self.trigger_img, (self.img_width - self.trigger_size, self.img_height - self.trigger_size))
        return img


class PoisonDataset(VisionDataset):

    def __init__(self, original_dataset: VisionDataset, target_class: int, trigger_path: str, 
                 transform : torch.nn.Module = None, trigger_size: int = 3, seed: int = None, poisoning_rate: float = 0.1):
        
        self.original_dataset = original_dataset
        self.transform = transform

        self.target_class = target_class
        self.poisoning_rate = poisoning_rate
        
        self.width, self.height, self.channels = original_dataset.data.shape[1:]
        self.trigger_handler = TriggerHandler(trigger_path, trigger_size, self.width, self.height)

        indices = [i for i in range(len(original_dataset.targets)) if original_dataset.targets[i]!=target_class]
        if seed: random.seed(seed)
        self.poi_indices = random.sample(indices, k=int(len(original_dataset.targets) * self.poisoning_rate))

    def __getitem__(self, index):
        img, original_label = self.original_dataset[index]
        # NOTE: According to the threat model, the trigger should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices:
            poison_label = self.target_class
            img = self.trigger_handler.put_trigger(img)
        else:
            poison_label = original_label

        if self.transform is not None:
            img = self.transform(img)

        return img, poison_label, original_label
    
    def __len__(self):
        return len(self.original_dataset)
    
