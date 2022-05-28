from matplotlib.image import imread
from utils import loader
from methods import hog
from methods import dimensionality_reduction as dr
import numpy as np
import random
import os
from PIL import Image

def main(args = None):

    # To keep random number generation consistent between runs
    random.seed(0)

    ############################################################################
    #                                                                          #
    # DATA LOADING                                                             #
    #                                                                          #
    ############################################################################
    # Load the images
    #rgbData = loader.loadImgs('../Dataset/sunny_day_sequence/')
    # Pick 80% of imgs as training data, the rest as test data
    #trainData = rgbData[:round(len(rgbData) * 0.8)]
    #testData = rgbData[round(len(rgbData) * 0.8):]

    ############################################################################
    #                                                                          #
    # FEATURE EXTRACTION WITH HISTOGRAM OF GRADIENTS                           #
    # (comment if using existing feature file)                                 #
    #                                                                          #
    ############################################################################
    
    # Process the training images - apply grayscale and resize for HOG
    #grayData = ip.grayscale(trainData)
    #hogData = ip.scale(grayData, 128, 64)

    # Extract the features from the training images using Histogram of Gradients
    #features = fe.hog(hogData)
    #np.save(os.path.join('./data', 'features'), features)
    
    ############################################################################
    #                                                                          #
    # DIMENSIONALITY REDUCTION WITH PRINCIPAL COMPONENT ANALYSIS               #
    ############################################################################
    # Load the feature vectors (comment if doing the extraction step)
    #features = np.load('data/features.npy')
    # Apply PCA
    # normalized = dr.pca(features)

    img = Image.open('../Dataset/Pedestrians/pos00003.pgm').convert('LA')
    hog.hog(img)



    #plt.figure()
    #f, axarr = plt.subplots(1,2) 
    #axarr[0].imshow(trainData[0])
    #axarr[1].imshow(grayData[0], cmap='gray')
    #plt.show()

if __name__ == "__main__":
    main()