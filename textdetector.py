# Code from: https://learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/
# Using this for reference as well: https://learnopencv.com/deep-learning-with-opencvs-dnn-module-a-definitive-guide/
# cd Documents\GitHub\MTG-Image-Titles-To-Text\

import cv2
import numpy as np


net = cv2.dnn.readNet(model="frozen_east_text_detection.pb")

# loading image
image = cv2.imread("CardPileSample1.jpg")
# making blob from image
inpWidth = 320
inpHeight = 320

blob = cv2.dnn.blobFromImage(image=image, scalefactor=1.0, size=(inpWidth, inpHeight), mean=(123.68, 116.78, 103.94))
'''
- The first argument is the image itself
- The second argument specifies the scaling of each pixel value. In this case, it is not required. Thus we keep it as 1.
- The default input to the network is 320×320. So, we need to specify this while creating the blob. You can experiment with any other input dimension, also.
- We also specify the mean that should be subtracted from each image since this was used while training the model. The mean used is (123.68, 116.78, 103.94).
- The next argument is whether we want to swap the R and B channels. This is required since OpenCV uses BGR format and Tensorflow uses RGB format.
- The last argument is whether we want to crop the image and take the center crop. We specify False in this case.
'''

# on Step 4: Forward Pass
# https://learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/

print("Done")