from methods.nms import nms
from skimage.feature import hog
from skimage.transform import pyramid_gaussian

class Detector():
    def __init__(self, classifier,
                       stepSize = [10, 10],
                       downscale = 1.5,
                       nmsThreshold = 0.5) -> None:
        self.clf = classifier
        self.mws = [64, 128]
        self.ss = stepSize
        self.ds = downscale
        self.nmsth = nmsThreshold

    def process(self, img):
        """Pyramid-HOG detection. A Gaussian Pyramid is generated from the 
        image. A sliding window is passed over each image in the pyramid, 
        extracting a HOG feature vector. The feature vector is classified with
        an SVM. Non-Maximum Suppression is applied to the resulting list of
        bounding boxes.

        Parameters
        ----------
        img : float matrix
            The image to be processed.

        Returns
        -------
        list, containing the filtered bounding boxes.
        """
        proposals = []
        # The current scale of the image
        scale = 0
        # Downscale the image and iterate
        for imgScaled in pyramid_gaussian(img, downscale=self.ds):
            # End if the sliding window is smaller than the minimum
            if imgScaled.shape[0] < self.mws[1] or imgScaled.shape[1] < self.mws[0]:
                break
            for (x, y, window) in self.sliding_window(imgScaled, self.mws, self.ss):
                # Skip if the window is smaller than the minimum
                if window.shape[0] != self.mws[1] or window.shape[1] != self.mws[0]:
                    continue
                # Calculate the HOG features and make a prediction
                f = hog(window, cells_per_block=(2,2)).reshape(1, -1) # Need to reshape cause it is only 1 sample
                prediction = self.clf.predict(f)
                if prediction[0] == 1:
                    # We found a pedestrian, append the bounding box to the list
                    # proposal: (top left corner coords, confidence score, width, height)
                    proposals.append((x, y, prediction[1],
                        int(self.mws[0]*(self.ds**scale)),
                        int(self.mws[1]*(self.ds**scale))))
            # Move the the next scale
            scale += 1
        # Perform Non Maximum Suppression
        return nms(proposals, self.nmsth)

    def sliding_window(self, image, window_size, step_size):
        """Generator function that yields a sliding window over an image.

        Parameters
        ----------
        img : float matrix
            The image to be processed.

        window_size : list of length 2
            Holds, in order, the width and height of the sliding window.

        window_size : list of length 2
            Holds, in order, the width and height of the sliding window.
    
        Yields
        ------
        float matrix, representing the next window read from the image.
        """
        for y in range(0, image.shape[0], step_size[1]):
            for x in range(0, image.shape[1], step_size[0]):
                yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])   