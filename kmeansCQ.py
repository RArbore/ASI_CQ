import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from ezras_kmeans import kmeans
import math


def pixelsForm(tensor): # Tensor(3, 256, 256) -> Tensor(65536, 3)
    return tensor.permute(1, 2, 0).view(256**2, 3)

def imageForm(tensor): # Tensor(65536, 3) -> Tensor(3, 256, 256)
    return tensor.view(256, 256, 3).permute(2, 0, 1)

def hex_to_I(hex): #hex -> [0,1]
    return int(hex,16)/256

def manual(imageTensor, manual_palette_input):
    # Parses the pallete into Tensor(# of colors, 3) of [0,1]
    palette = torch.zeros(len(manual_palette_input),3)
    for color in manual_palette_input:
        palette[manual_palette_input.index(color)] = torch.Tensor([hex_to_I(color[1:3]), hex_to_I(color[3:5]), hex_to_I(color[5:])])
    # Finds the closest color in the palette for each pixel
    imageTensor = pixelsForm(imageTensor)
    palette_ids = torch.zeros(len(imageTensor))
    for pixel in range(len(imageTensor)):
        closest_id, closest = (0, math.inf)
        for color_id in range(len(palette)):
            d = sum((palette[color_id]-imageTensor[pixel])**2)
            if d < closest:
                closest_id, closest = color_id, d
        #import pdb; pdb.set_trace()
        palette_ids[pixel] = torch.Tensor([closest_id])
    return palette_ids, palette #palette_ids = Tensor(65536), palette = Tensor(# of colors)

def pixel_mapping(mapTensor, paletteTensor):
    return torch.stack([imageForm(torch.stack([paletteTensor[imageNum][pixel] for pixel in mapTensor[imageNum].int()])) for imageNum in range(mapTensor.size(0))]) #returns Tensor(# of images, 3, 256, 256)

def validLoss(original, quantized):
    return float(torch.mean((original-quantized)**2))


# Setting parameters
num_clusters = 64
iteration_limit = 1000
tolerance = 0

PVsAfterDecimal = 10
saveResults = True

# Converting image file(s) into Tensor([# of files, 3, 256, 256])
imagesList = []
path = 'testImages/'
directory = os.listdir(path)
for file in directory:
    imagesList.append(transforms.ToTensor()(Image.open(path+file)))
imagesTensor = pad_sequence(imagesList).permute(1,0,2,3) #Tensor([# of files, 3, 256, 256])

# Initializing (mapTensor, paletteTensor) with previous results if they exist, or zeros
mapTensor = torch.zeros(imagesTensor.size(0), 256**2)
paletteTensor = torch.ones(imagesTensor.size(0), num_clusters, 3)
startingIndex = 0
if os.path.exists('saved.pt'):
    userInput = ''
    while not userInput in ['c','r']:
        userInput = input('There are quantized images saved from the previous run.  Continue working where the previous run left off(c) or reset(r)?\n>>> ')
    if userInput == 'c':
        mapTensor, paletteTensor = torch.load('saved.pt')
        if len(torch.nonzero(paletteTensor[imagesTensor.size(0)-1] == 1)) == 3*num_clusters: #if the last image is all ones still
            startingIndex = min([i for i in range(mapTensor.size(0)) if mapTensor[i].unique().tolist() == [0]])
        else:
            startingIndex = imagesTensor.size(0)
    else:
        os.remove('saved.pt')

# Performing chosen method to get mappings and palette
for image in range(startingIndex, imagesTensor.size(0)):
    PFimage = pixelsForm(imagesTensor[image])
    mapTensor[image], paletteTensor[image] = kmeans(X=PFimage, num_clusters=num_clusters, distance='euclidean', iteration_limit=iteration_limit, tol=tolerance, image=image)
    if saveResults:
        torch.save((mapTensor, paletteTensor), 'saved.pt') #mapTensor = Tensor(# of images, 256**2), paletteTensor = Tensor(images, # of colors)
    quantizedImagesTensor = pixel_mapping(mapTensor[image].unsqueeze(0), paletteTensor[image].unsqueeze(0)).squeeze().squeeze(0)
    print('loss for image %s: %s\n' % (image, round(validLoss(imagesTensor[image], quantizedImagesTensor), PVsAfterDecimal)))
print('finished k-means for all images. Calculating loss...')
# imageTensor = imagesTensor[0] #Tensor(3, 256, 256)
# torch.save(manual(imageTensor,['#101918','#294c3c','#1b3441','#284b66','#576e48','#baab47','#39341d','#604d23','#5c89ae','#d0e1d3']), 'manual.pt')
# palette_ids, palette = torch.load('manual.pt')
# print(palette,palette_ids)

#Pixel mapping and calcuating validation loss
quantizedImages = pixel_mapping(mapTensor, paletteTensor)
batchValidLoss = validLoss(imagesTensor, quantizedImages)
print('validation loss for entire batch: %s\n' % (round(batchValidLoss,PVsAfterDecimal)))

#Plotting Quantized Images
if input('display images (y/n)?') == 'y':
    plt.tight_layout()
    for quantizedImage in range(len(quantizedImages)):
        plt.imshow((quantizedImages[quantizedImage].permute(1, 2, 0).numpy()*255).astype(np.uint8))
        plt.show() #imshow puts stff on the plot, show actually displays the plot
        input(f'displaying quantized image {quantizedImage} (enter for next)')
