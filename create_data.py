from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import os

dirs = ["imagesPlaces205_resize/data/vision/torralba/deeplearning/images256/a/aquarium"]

for dir in dirs:
    tensor_list = []
    for jpg in os.listdir(dir):
        if jpg.endswith(".jpg"):
            pil_img = Image.open(dir+"/"+jpg)
            tensor = transforms.ToTensor()(pil_img)
            if (tensor.size(0) == 1):
                tensor = torch.cat((tensor, tensor, tensor), dim=0)
            tensor_list.append(tensor)

    data = torch.stack(tensor_list).float().detach()
    print(data.size())
    torch.save(data, "TRAIN_"+dir.split("/")[7]+".pt")
