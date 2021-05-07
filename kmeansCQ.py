import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
# from ezras_kmeans import kmeans
from fast_kmeans import kmeans
from fast_kmeans import pairwise_distance
import math
import time

def pixelsForm(tensor, imagesSize): # Tensor(3, 256, 256) -> Tensor(65536, 3)
    return tensor.permute(1, 2, 0).reshape(np.prod(imagesSize), 3)

def imageForm(tensor, imagesSize): # Tensor(65536, 3) -> Tensor(3, 256, 256)
    return tensor.view(imagesSize[0], imagesSize[1], 3).permute(2, 0, 1)

def hex_to_I(hex): #hex -> [0,1]
    return int(hex,16)/256

def manual(imageTensor, manual_palette_input):
    # Parses the pallete into Tensor(# of colors, 3) of [0,1]
    palette = torch.zeros(len(manual_palette_input),3)
    for color in manual_palette_input:
        palette[manual_palette_input.index(color)] = torch.Tensor([hex_to_I(color[1:3]), hex_to_I(color[3:5]), hex_to_I(color[5:])])
    # Finds the closest color in the palette for each pixel
    imageTensor = pixelsForm(imageTensor, (256,256))
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

def pixel_mapping(imageTensor, paletteTensor):
    # return torch.stack([imageForm(torch.stack([paletteTensor[imageNum][pixel] for pixel in mapTensor[imageNum].int()]), imagesSize) for imageNum in range(mapTensor.size(0))]) #returns Tensor(# of images, 3, 256, 256)
    dis = pairwise_distance(imageTensor, paletteTensor)
    closest_cluster = torch.argmin(dis, dim=1)
    one_hot = (closest_cluster.unsqueeze(dim=1) == torch.arange(paletteTensor.size(0)).reshape(1, paletteTensor.size(0))).float()
    image = torch.stack((torch.matmul(one_hot.float(), paletteTensor[:,0].reshape(-1,1)),
                         torch.matmul(one_hot.float(), paletteTensor[:,1].reshape(-1,1)),
                         torch.matmul(one_hot.float(), paletteTensor[:,2].reshape(-1,1))),dim=2).squeeze()
    return image

def validLoss(original, quantized):
    return float(torch.mean((original-quantized)**2))

current_milli_time = lambda: int(round(time.time() * 1000))

def logTime(logString, time_rec):
    seconds = math.floor((time_rec) / 1000)
    minutes = math.floor(seconds / 60)
    seconds = seconds % 60
    return("%s %s minute(s) %s second(s)." % (logString, str(minutes), str(seconds)))

def preprocess(image):
    imagePalette = []
    frequencyTensor = []
    sortedImagePalette = sorted(image.tolist())
    current = []
    for index in range(len(sortedImagePalette)):
        if current == sortedImagePalette[index]:
            frequencyTensor[-1] += 1
        else:
            imagePalette.append(sortedImagePalette[index])
            frequencyTensor.append(1)
            current = sortedImagePalette[index]
    #REMOVE THIS FIRST RETURN VALUE:
    return torch.Tensor(sortedImagePalette).to(device), torch.Tensor(imagePalette).to(device), torch.Tensor(frequencyTensor).to(device)

# Setting parameters
imagesDirectoryPath = 'testImages/'

num_clusters = 64
iteration_limit = 1000
tolerance = 0

PVsAfterDecimal = 10
saveResults = False

# imagesTensor = torch.clamp(torch.load("TRAIN_aquarium.pt"), 0, 1)[15000:15100]

# Converting image file(s) into Tensor([# of files, 3, 256, 256])
imagesList = []
directory = os.listdir(imagesDirectoryPath)
imagesSize = (256,256)
for file in [i for i in directory if i != '.DS_Store']:
    with Image.open(imagesDirectoryPath + file) as image:
        if imagesSize == (0,0):
            imagesSize = (image.size[1],image.size[0])  #Why is it rotated?!?!?!
        elif imagesSize != (image.size[1],image.size[0]): #Why is it rotated?!?!?!
            print('Error: images in directory not the same size')
            exit()
        imagesList.append(transforms.ToTensor()(image))
imagesTensor = pad_sequence(imagesList).permute(1,0,2,3) #Tensor([# of files, 3, 256, 256])

# Initializing (mapTensor, paletteTensor) with previous results if they exist, or empty
mappings = []
paletteTensor = torch.ones(imagesTensor.size(0), num_clusters, 3)
startingIndex = 0
losses = [0 for _ in range(imagesTensor.size(0))]
if os.path.exists('saved.pt'):
    userInput = ''
    while not userInput in ['c','r']:
        userInput = input('There are quantized images saved from the previous run.  Continue working where the previous run left off(c) or reset(r)?\n>>> ')
    if userInput == 'c':
        quantizedImagesTensor, losses = torch.load('saved.pt')
        if len(torch.nonzero(paletteTensor[imagesTensor.size(0)-1] == 1)) == 3*num_clusters: #if the last image is all ones still
            startingIndex = min([i for i in range(mappings.size(0)) if mappings[i].unique().tolist() == [0]])
        else:
            startingIndex = imagesTensor.size(0)
    else:
        os.remove('saved.pt')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
time_sum = 0.0
iters = 0.0
# Performing chosen method to get mappings and palette
quantizedImagesTensor = torch.zeros(imagesTensor.size())
for image in range(startingIndex, imagesTensor.size(0)):
    before_time = current_milli_time()
    PFimage, imagePalette, frequencyTensor = preprocess(pixelsForm(imagesTensor[image], imagesSize))
    map, paletteTensor[image] = kmeans(fast_kmeans = True, X = PFimage, imagePalette=imagePalette, frequencyTensor=frequencyTensor, num_clusters=num_clusters, iteration_limit=iteration_limit, tol=tolerance, image=image, device=device)
    time_rec = current_milli_time() - before_time
    print('loss: %s\n%s\n' % (round(losses[image], PVsAfterDecimal), logTime('Total time:', time_rec)))
    time_sum += time_rec
    iters += 1.0
    mappings.append(map)
    quantizedImagesTensor[image] = imageForm(pixel_mapping(pixelsForm(imagesTensor[image], imagesSize), paletteTensor[image]), imagesSize)
    losses[image] = (validLoss(imagesTensor[image], quantizedImagesTensor[image]))
    if saveResults:
        torch.save((quantizedImagesTensor, losses), 'saved.pt') #quantizedImagesTensor = tensor(# of files, 3, 256, 256), losses = list of length: # of files

print(time_sum/iters, "seconds on average")
#print('finished k-means for all images. Calculating loss...')
# imageTensor = imagesTensor[0] #Tensor(3, 256, 256)
# torch.save(manual(imageTensor,['#101918','#294c3c','#1b3441','#284b66','#576e48','#baab47','#39341d','#604d23','#5c89ae','#d0e1d3']), 'manual.pt')
# palette_ids, palette = torch.load('manual.pt')
# print(palette,palette_ids)

#Pixel mapping and calcuating validation loss
#print('validation loss for entire batch: %s\n%s\n' % (round(sum(losses)/len(losses),PVsAfterDecimal),logTime('Average time to complete quantization for each images:', time_sum / iters)))

#Plotting Quantized Images
#if input('display images (y/n)?') == 'y':
#    plt.tight_layout()
#    for quantizedImage in range(len(quantizedImagesTensor)):
#        plt.imshow((quantizedImagesTensor[quantizedImage].permute(1, 2, 0).numpy()*255).astype(np.uint8))
#        plt.show() #imshow puts stff on the plot, show actually displays the plot
#        input(f'displaying quantized image {quantizedImage} (enter for next)')
