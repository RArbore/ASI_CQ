from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import os

tensor_list = []

dir = "jpgs/a/aquarium"
for jpg in os.listdir(dir):
    if jpg.endswith(".jpg"):
        pil_img = Image.open(dir+"/"+jpg)
        tensor = transforms.ToTensor()(pil_img)
        if (tensor.size(0) == 1):
            tensor = torch.cat((tensor, tensor, tensor), dim=0)
        tensor_list.append(tensor)

# print(tensor_list[0].size())
# torch.save(tensor_list[0], "what.pt")
# print("?")
# torch.save(torch.cat((tensor_list[0], tensor_list[1]), dim=0), "what.pt")
# print("?!")

# data = tensor_list[0].view(1, 3, 256, 256)
# for i in range(1, 100):
#     data = torch.cat((data, tensor_list[i].view(1, 3, 256, 256)))
# print(data.size())
# torch.save(data, "what.pt")
# print("excuse me")

data = torch.stack(tensor_list).float().detach()
print(data.size())
torch.save(data, "TRAIN.pt")
