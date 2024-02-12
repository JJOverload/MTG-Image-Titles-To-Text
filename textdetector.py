# Code from: https://learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/
# Using this for reference as well: https://learnopencv.com/deep-learning-with-opencvs-dnn-module-a-definitive-guide/
# https://github.com/spmallick/learnopencv/blob/master/TextDetectionEAST/textDetection.py
# cd Documents\GitHub\MTG-Image-Titles-To-Text\

import cv2
import numpy as np

def decode(scores, geometry, scoreThresh):
    detections = []
    confidences = []

    ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):

        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]

            # If score is lower than threshold score, move to next x
            if(score < scoreThresh):
                continue

            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            # Find points for rectangle
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
            center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
            detections.append((center, (w,h), -1*angle * 180.0 / math.pi))
            confidences.append(float(score))

    # Return detections and confidences
    return [detections, confidences]


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

outputLayers = []
outputLayers.append("feature_fusion/Conv_7/Sigmoid")
outputLayers.append("feature_fusion/concat_3")

net.setInput(blob)
output = net.forward(outputLayers)
scores = output[0]
geometry = output[1]

[boxes, confidences] = decode(scores, geometry, confThreshold)


print("Done")