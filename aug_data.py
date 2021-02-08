from torchvision import transforms
import elasticdeform
import random
import resnet
import torch
import time
import math
import sys
import os

NUM_AUGS = 3

train_data_load = torch.load("TRAIN_aquarium.pt") #(15100, 3, 256, 256)
train_data_load = torch.clamp(train_data_load[torch.randperm(train_data_load.size(0))], 0, 1)
valid_data = train_data_load[15000:15100]
train_data = train_data_load[0:15000]
for i in range(NUM_AUGS):
    aug = torch.from_numpy(elasticdeform.deform_random_grid(train_data_load[0:15000].numpy(), sigma=0.8, points=3, axis=(2, 3)))
    train_data = torch.cat((train_data, aug), dim=0)
    print("Augmentation done : "+str(i+1))
print(train_data.size())
train_data = torch.cat((train_data, valid_data), dim=0)
print(train_data.size())
torch.save(train_data, "AUG_aquarium.pt")