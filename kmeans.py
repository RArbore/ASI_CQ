#Check if the sse is 0 for all possible colors in manual pallette, check displaying regular image to see why 3x3 grid of images

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from kmeans_pytorch import kmeans
import math

def convert(hex): #hex -> [0,1]
    return int(hex,16)/256

def distance(a, b):
    return sum((a-b)**2)

def pixelsForm(tensor): # Tensor(3, 256, 256) -> Tensor(65536, 3)
    return tensor.permute(1, 2, 0).view(256**2, 3)

def imageForm(tensor): # Tensor(65536, 3) -> Tensor(3, 256, 256)
    return tensor.view(256, 256, 3).permute(2, 0, 1)

def manual(imageTensor, manual_palette_input):
# Parses the pallete into Tensor(# of colors, 3) of [0,1]
    palette = torch.zeros(len(manual_palette_input),3)
    for color in manual_palette_input:
        palette[manual_palette_input.index(color)] = torch.Tensor([convert(color[1:3]), convert(color[3:5]), convert(color[5:])])
# Finds the closest color in the palette for each pixel
    imageTensor = pixelsForm(imageTensor)
    palette_ids = torch.zeros(len(imageTensor))
    for pixel in range(len(imageTensor)):
        closest_id, closest = (0, math.inf)
        for color_id in range(len(palette)):
            d = distance(palette[color_id], imageTensor[pixel])
            if d < closest:
                closest_id, closest = color_id, d
        #import pdb; pdb.set_trace()
        palette_ids[pixel] = torch.Tensor([closest_id])
    return palette_ids, palette #palette_ids = Tensor(65536), palette = Tensor(# of colors)

def k_means(imagesTensor):
    num_clusters = 64
    imagesTensor = pixelsForm(imagesTensor)
    return kmeans(X=imagesTensor, num_clusters=num_clusters, distance='euclidean')

def quantize(imageTensor, palette, palette_ids):
    mappingsTensor = pixelsForm(imageTensor)
    SSE = 0
    for pixel_id in range(mappingsTensor.size()[0]):
        SSE += distance(mappingsTensor[pixel_id], palette[int(palette_ids[pixel_id])]) / 256**2
        mappingsTensor[pixel_id] = palette[int(palette_ids[pixel_id])]
    return imageForm(mappingsTensor), SSE #returns Tensor(3, 256, 256), int

# Converting image file(s) into Tensor([# of files, 3, 256, 256])
imagesList = []
path = 'testImages/'
directory = os.listdir(path)
for file in directory:
    imagesList.append(transforms.ToTensor()(Image.open(path+file)))
imagesTensor = pad_sequence(imagesList).permute(1,0,2,3) #Tensor([# of files, 3, 256, 256])

#Performing chosen method to get mappings and palette
imageTensor = imagesTensor[0] #Tensor(3, 256, 256)
# torch.save(manual(imageTensor,['#101918','#294c3c','#1b3441','#284b66','#576e48','#baab47','#39341d','#604d23','#5c89ae','#d0e1d3']), 'manual.pt')
# palette_ids, palette = torch.load('manual.pt')
palette_ids, palette = k_means(imageTensor)
print('Created mappings and palette')
# print(palette,palette_ids)

#Quantizing the image
quantizedImage, SSE = quantize(imageTensor, palette, palette_ids)
print('Quantized image\nSSE: %s' % SSE)

plt.imshow((quantizedImage.permute(1, 2, 0).numpy()*255).astype(np.uint8))
plt.tight_layout()
plt.show() #imshow puts stff on the plot, show actually displays the plot

#for pixel in each image, find the minimum value of (red-palette_red^2, green-palette_green^2, blue-palette_blue^2)