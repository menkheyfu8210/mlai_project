import cv2
import time

from methods.classifiers import SVM, KNN, ParzenWindows
from methods.detector import Detector
from methods.nms import nms
from methods.feature_extractor import extractFeatures, loadFeatures
from skimage.transform import pyramid_gaussian, resize
from skimage.io import imread
from skimage.feature import hog

def sliding_window(image, window_size, step_size):
    '''
    This function returns a patch of the input image `image` of size equal
    to `window_size`. The first image returned top-left co-ordinates (0, 0) 
    and are increment in both x and y directions by the `step_size` supplied.
    So, the input parameters are -
    * `image` - Input Image
    * `window_size` - Size of Sliding Window
    * `step_size` - Incremented Size of Window
    The function returns a tuple -
    (x, y, im_window)
    where
    * x is the top-left x co-ordinate
    * y is the top-left y co-ordinate
    * im_window is the sliding window image
    '''
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def main(args = None):

    preExtractedFeatures = True

    if not preExtractedFeatures:
        extractFeatures()
    trainFeatures = []
    trainLabels = []
    loadFeatures(trainFeatures, trainLabels)

    #svm = SVM(True, './models/linearSVM.mod')
    #knn = KNN(10, 'euclidean', trainFeatures, trainLabels)
    pw = ParzenWindows(0.2, 'gaussian', trainFeatures)

    """testFeatures = []
    testLabels = []
    loadFeatures(testFeatures, testLabels, True)
    predicted = svm.model.predict(testFeatures)
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
    accuracy = accuracy_score(testLabels,predicted)
    precision_np = precision_score(testLabels,predicted,pos_label=0)
    precision_p = precision_score(testLabels,predicted,pos_label=1)
    recall_np = recall_score(testLabels,predicted,pos_label=0)
    recall_p = recall_score(testLabels,predicted,pos_label=1)
    print('Classifier accuracy: ' + "{0:.2f}".format(accuracy*100) + '%')
    print('Precision w.r.t class non-pedestrian: ' + "{0:.2f}".format(precision_np))
    print('Precision w.r.t class pedestrian: ' + "{0:.2f}".format(precision_p))
    print('Recall w.r.t class non-pedestrian: ' + "{0:.2f}".format(recall_np))
    print('Recall w.r.t class pedestrian: ' + "{0:.2f}".format(recall_p))
    cfs = confusion_matrix(testLabels, predicted)
    print(cfs)"""

    imgs = [imread('./test/test1.png', as_gray=True),
            imread('./test/test2.png', as_gray=True),
            imread('./test/test3.png', as_gray=True),
            imread('./test/test4.png', as_gray=True)]
            
    # 480 640
    for im in imgs:
        start_time = time.time()
        im = resize(im, (240, 320))

        detector = Detector(pw,
                        stepSize = [6,6],
                        downscale = 1.25,
                        nmsThreshold = 0.05)
        detections = detector.process(im)

        # Display the results after performing NMS
        for (x_tl, y_tl, s, w, h) in detections:

            # Draw the detections
            cv2.rectangle(im, (x_tl, y_tl), (x_tl + w, y_tl + h), (0,0,0))
            #cv2.putText(im, "%.2f" % s, (x_tl, y_tl - 10), cv2.FONT_HERSHEY_PLAIN, 1.1, (0,0,0), 2, cv2.LINE_AA)

        print(f"elapsed:{time.time() - start_time}")
        cv2.imshow("Detected pedestrians", im)
        cv2.waitKey()

if __name__ == "__main__":
    main()