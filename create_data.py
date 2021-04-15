from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import tqdm
import os

dirs = ["imagesPlaces205_resize/data/vision/torralba/deeplearning/images256/a/abbey",
        "imagesPlaces205_resize/data/vision/torralba/deeplearning/images256/a/airport_terminal",
        "imagesPlaces205_resize/data/vision/torralba/deeplearning/images256/a/alley",
        "imagesPlaces205_resize/data/vision/torralba/deeplearning/images256/a/amphitheater",
        "imagesPlaces205_resize/data/vision/torralba/deeplearning/images256/a/amusement_park",
        "imagesPlaces205_resize/data/vision/torralba/deeplearning/images256/a/apartment_building",
        "imagesPlaces205_resize/data/vision/torralba/deeplearning/images256/a/aquarium",
        "imagesPlaces205_resize/data/vision/torralba/deeplearning/images256/a/aqueduct",
        "imagesPlaces205_resize/data/vision/torralba/deeplearning/images256/a/arch",
        "imagesPlaces205_resize/data/vision/torralba/deeplearning/images256/a/art_gallery",
        "imagesPlaces205_resize/data/vision/torralba/deeplearning/images256/a/art_studio",
        "imagesPlaces205_resize/data/vision/torralba/deeplearning/images256/a/assembly_line",
        "imagesPlaces205_resize/data/vision/torralba/deeplearning/images256/a/attic",
        "imagesPlaces205_resize/data/vision/torralba/deeplearning/images256/a/auditorium"]

for dir in dirs:
    tensor_list = []
    for jpg in tqdm.tqdm(os.listdir(dir)):
        if jpg.endswith(".jpg"):
            pil_img = Image.open(dir+"/"+jpg)
            tensor = transforms.ToTensor()(pil_img)
            if (tensor.size(0) == 1):
                tensor = torch.cat((tensor, tensor, tensor), dim=0)
            tensor_list.append(tensor)

    data = torch.stack(tensor_list).float().detach()
    print(data.size())
    torch.save(data, "TRAIN_"+dir.split("/")[7]+".pt")
