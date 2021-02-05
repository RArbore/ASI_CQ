import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from kmeans_pytorch import kmeans
import math

#images -> tensor

imagesList = []
path = 'testImages/'
directory = os.listdir(path)
for file in directory:
    imagesList.append(transforms.ToTensor()(Image.open(path+file)))
imagesTensor = pad_sequence(imagesList).permute(1,0,2,3) #tensor size: [# of files, 3, 256, 256]
#print(imagesTensor.size())
imageTensor = imagesTensor[0].reshape(256**2,3) #only using the first image and collapsing the last two dimensions

def convert(hex): #hex -> [0,1]
    return int(hex,16)/256

def distance(a, b):
    return sum((a-b)**2)

def manual(manual_palette_input):
    palette = torch.zeros(len(manual_palette_input),3)
    #import pdb; pdb.set_trace()
    for color in manual_palette_input:
        palette[manual_palette_input.index(color)] = torch.Tensor([convert(color[1:3]), convert(color[3:5]), convert(color[5:])])
    palette_ids = torch.zeros(len(imageTensor))
    for pixel in range(len(imageTensor)):
        closest_id, closest = (0, math.inf)
        for color_id in range(len(palette)):
            d = distance(palette[color_id], imageTensor[pixel])
            if d < closest:
                closest_id, closest = color_id, d
        #import pdb; pdb.set_trace()
        palette_ids[pixel] = torch.Tensor([closest_id])
    return palette_ids, palette

def kmeans():
    num_clusters = 64
    return kmeans(X=imageTensor, num_clusters=num_clusters, distance='euclidean')


#torch.save(manual(['#284168','#19243e','#283227','#1f5a55','#455622','#6a95af','#17271d','#fffb9e','#9d852b','#1f2b42','#1f2b42','#604d23','#294c3c','#d0e1d3']), 'manual.pt')
palette_ids, palette = torch.load('manual.pt')
print(palette,palette_ids)

SSE = 0
quantizedImage = imageTensor.clone()
for pixel_id in range(imageTensor.size()[0]):
    SSE += distance(imageTensor[pixel_id], palette[int(palette_ids[pixel_id])])
    quantizedImage[pixel_id] = palette[int(palette_ids[pixel_id])]
print(SSE)


plt.imshow((quantizedImage.view(256,256,3).numpy()*255).astype(np.uint8))
plt.tight_layout()
plt.show() #imshow puts stff on the plot, show actually displays the plot

#for pixel in each image, find the minimum value of (red-palette_red^2, green-palette_green^2, blue-palette_blue^2)