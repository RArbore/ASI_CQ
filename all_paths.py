import os

header = "imagesPlaces205_resize/data/vision/torralba/deeplearning/images256/"
dirs = os.listdir(header)
f = open("paths.txt", "a")

for dir in dirs:
    scenes = os.listdir(header+dir)
    for scene in scenes:
        paths = os.listdir(header+dir+"/"+scene)
        for path in paths:
            f.write(dir+"/"+scene+"/"+path+"\n")
f.close()
