import math
from skimage.feature import hog
from skimage import data, exposure

class FeatureExtract:
      def __init__(self,img):
          self.imge = img
      def features(self,imge):
              fd,hog_image = hog(imge, orientations=16, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualize=True)
              hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
              return hog_image
              