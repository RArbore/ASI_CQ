from torchvision import transforms
from PIL import Image
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
data = torch.stack(tensor_list).float()
print(data.size())
torch.save(data, "TRAIN.pt")
