from torchvision import transforms
from PIL import Image
import numpy as np
import random
import torch
import tqdm
import os

NUM_PATHS = 2223402
DATA_SIZE = 15100

f = open("paths.txt", "r")
paths = f.read().split("\n")[:-1]
tensor_list = []
length = len(paths)

for i in tqdm.tqdm(range(DATA_SIZE)):
    length -= 1
    choice_i = random.randint(0, length)
    choice = paths[choice_i]
    paths.pop(choice_i)

    pil_img = Image.open("images256/"+choice)
    tensor = transforms.ToTensor()(pil_img)
    if (tensor.size(0) == 1):
        tensor = torch.cat((tensor, tensor, tensor), dim=0)
    tensor_list.append(tensor)

if tensor_list:
    data = torch.stack(tensor_list).float().detach()
    print(data.size())
    torch.save(data, "TRAIN_DATA.pt")
