import cv2 as cv
import time
import glob
import joblib
import os

from skimage.feature import hog
from skimage.color import rgb2gray

path = './Project/dataset/DC-ped-dataset_base/1/non-ped_examples/img_00000.pgm'

def main():
    scaler = joblib.load('./Project/features/scaler.mod')
    pca = joblib.load('./Project/features/pca.mod')
    start = time.time()
    img = cv.imread(path)
    img = rgb2gray(img)
    img = cv.resize(img, (64, 128))
    f = hog(img, cells_per_block=(2,2)).reshape(1, -1)
    f = scaler.transform(f)
    f = pca.transform(f)
    end = time.time()
    print('elapsed: ' + str(end - start) + 's')
    h = cv.HOGDescriptor()
    start = time.time()
    img = cv.imread(path)
    img = rgb2gray(img)
    img = cv.resize(img, (64, 128))
    f = h.compute(img)
    f = scaler.transform(f)
    f = pca.transform(f)
    end = time.time()
    print('elapsed: ' + str(end - start) + 's')

if __name__ == '__main__':
    main()