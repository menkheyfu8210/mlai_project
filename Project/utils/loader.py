import os
from os.path import isfile, join
from PIL import Image
import random

def loadImgs(dataPath, n=0):
    # Get list of .pgm files in dataPath directory
    files = [f for f in os.listdir(dataPath) if isfile(join(dataPath, f)) and f.endswith('.pgm')]
    if n != 0:
        files = random.sample(files, n)
    res = []
    # Convert each image to grayscale and append to res
    for f in files:
        img = Image.open(os.path.join(dataPath, f)).convert('LA')
        res.append(img)
    return res
