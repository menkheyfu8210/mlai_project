import numpy as np
import os

from os.path import isfile, join
from PIL import Image
from sklearn.feature_extraction.image import extract_patches_2d

def main(args = None):
    # Extract 8 random 64x128 patches from the non-pedestrian dataset
    files = [f for f in os.listdir('../Dataset/NonPedestrians/') if isfile(join('../Dataset/NonPedestrians/', f)) and f.endswith('.pgm')]
    negData = [np.array(Image.open(os.path.join('../Dataset/NonPedestrians/', f))) for f in files]
    i = 0
    for n in negData:
        patches = extract_patches_2d(n, [128, 64], max_patches=6)
        for p in patches:
            im = Image.fromarray(p)
            fp = '../Dataset/NonPedestrianPatches/neg' + "%05d" % (i,) + '.pgm'
            im.save(fp)
            i = i + 1

if __name__ == "__main__":
    main()