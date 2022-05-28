import os
from os.path import isfile, join
import matplotlib.image as im

def loadImgs(dataPath):
    # Get list of .png files in dataPath directory
    files = [f for f in os.listdir(dataPath) if isfile(join(dataPath, f)) and f.endswith('.png')]
    res = []
    # Convert each image to grayscale and append to res
    for f in files:
        img = im.imread(dataPath + f)
        res.append(img)
    return res
