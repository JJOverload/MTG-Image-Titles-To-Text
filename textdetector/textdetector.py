# Code from: https://learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/
# Using this for reference as well: https://learnopencv.com/deep-learning-with-opencvs-dnn-module-a-definitive-guide/
# https://github.com/spmallick/learnopencv/blob/master/TextDetectionEAST/textDetection.py
# https://pyimagesearch.com/2021/01/19/image-masking-with-opencv/

# Inspired by this code for applying merging of bounding boxes algorithm:
# https://stackoverflow.com/questions/55593506/merge-the-bounding-boxes-near-by-into-one

# Sample CMD commands:
# cd Documents\GitHub\MTG-Image-Titles-To-Text\textdetector
# python textdetector.py --input CardPileSample1.jpg --width 3072 --height 4096
# python textdetector.py --input tegwyll-nonlands.jpg --width 3072 --height 4064
# python textdetector.py --input tegwyll-nonlands-Copy.jpg --width 3072 --height 2656
# python textdetector.py --input 1_python-ocr.jpg --width 800 --height 352

# Import for text detection
import cv2 as cv
import math
import argparse

# Import for rectangle
import numpy as np

# Import for OSD (pytesseract)
from PIL import Image
import pytesseract

parser = argparse.ArgumentParser(description='Use this script to run text detection deep learning networks using OpenCV.')
# Input argument
parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
# Model argument
parser.add_argument('--model', default="frozen_east_text_detection.pb",
                    help='Path to a binary .pb file of model contains trained weights.'
                    )
# Width argument
parser.add_argument('--width', type=int, default=320,
                    help='Preprocess input image by resizing to a specific width. It should be multiple by 32.'
                   )
# Height argument
parser.add_argument('--height',type=int, default=320,
                    help='Preprocess input image by resizing to a specific height. It should be multiple by 32.'
                   )
# Confidence threshold
parser.add_argument('--thr',type=float, default=0.5,
                    help='Confidence threshold.'
                   )
# Non-maximum suppression threshold
parser.add_argument('--nms',type=float, default=0.4,
                    help='Non-maximum suppression threshold.'
                   )

parser.add_argument('--device', default="cpu", help="Device to inference on")


args = parser.parse_args()


############ Utility functions ############
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

def merge_boxes(box1, box2):
    box1_xmin, box1_ymin, box1_xmax, box1_ymax = box1
    box2_xmin, box2_ymin, box2_xmax, box2_ymax = box2

    return( (min(box1_xmin, box2_xmin), min(box1_ymin, box2_ymin), max(box1_xmax, box2_xmax), max(box1_ymax, box2_ymax)) )

def calc_sim(text, obj):
    # text: xmin, ymin, xmax, ymax
    # obj: xmin, ymin, xmax, ymax
    text_xmin, text_ymin, text_xmax, text_ymax = text
    obj_xmin, obj_ymin, obj_xmax, obj_ymax = obj

    x_dist = min(abs(text_xmin-obj_xmin), abs(text_xmin-obj_xmax), abs(text_xmax-obj_xmin), abs(text_xmax-obj_xmax))
    y_dist = min(abs(text_ymin-obj_ymin), abs(text_ymin-obj_ymax), abs(text_ymax-obj_ymin), abs(text_ymax-obj_ymax))

    dist = x_dist + y_dist
    return(dist)

def merge_algo(bboxes):
    for j in bboxes:
        for k in bboxes:
            if j == k: #continue on if we are comparing a box with itself
                continue
            # Find out if these two bboxes are within distance limit
            if calc_sim(j, k) < dist_limit:
                # Create a new box
                new_box = merge_boxes(j, k)
                bboxes.append(new_box)
                # Remove previous boxes
                bboxes.remove(j)
                bboxes.remove(k)

                #Return True and new "bboxes"
                return(True, bboxes)
    return(False, bboxes)



if __name__ == "__main__":
    # Read and store arguments
    confThreshold = args.thr
    nmsThreshold = args.nms
    inpWidth = args.width
    inpHeight = args.height
    model = args.model

    # Creating buffer/container for bounding boxes' vertices
    bbox = []
    # Setting distance limit for bounding box merging
    dist_limit = 40

    # Load network
    net = cv.dnn.readNet(model)
    if args.device == "cpu":
        net.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
        print("Using CPU device")
    elif args.device == "gpu":
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        print("Using GPU device")


    # Create a new named window
    kWinName = "EAST: An Efficient and Accurate Scene Text Detector"
    cv.namedWindow(kWinName, cv.WINDOW_NORMAL)
    outputLayers = []
    outputLayers.append("feature_fusion/Conv_7/Sigmoid")
    outputLayers.append("feature_fusion/concat_3")

    # Open a video file or an image file or a camera stream
    cap = cv.VideoCapture(args.input if args.input else 0)

    while cv.waitKey(1) < 0:
        # Read frame
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break

        # Get frame height and width
        height_ = frame.shape[0]
        width_ = frame.shape[1]
        rW = width_ / float(inpWidth)
        rH = height_ / float(inpHeight)

        # Create a 4D blob from frame.
        blob = cv.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)

        # Run the model
        net.setInput(blob)
        output = net.forward(outputLayers)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())

        
        # have buffers for getting coordinates for rectangles
        startCorner = (0, 0)
        endCorner = (0, 0)
        #creating mask layer to work on later...
        mask = np.zeros(frame.shape[:2], dtype="uint8")
        mask2 = np.zeros(frame.shape[:2], dtype="uint8")
        #creating backup frame
        frame2 = frame.copy()

        # Get scores and geometry
        scores = output[0]
        geometry = output[1]
        [boxes, confidences] = decode(scores, geometry, confThreshold)
        # Apply NMS
        indices = cv.dnn.NMSBoxesRotated(boxes, confidences, confThreshold, nmsThreshold) 
        for i in indices:
            wlist = []
            hlist = []
            # get 4 corners of the rotated rect
            vertices = cv.boxPoints(boxes[i])
            # scale the bounding box coordinates based on the respective ratios
            for j in range(4):
                vertices[j][0] *= rW
                vertices[j][1] *= rH
                #populating list for min/max getting and text isolation
                wlist.append(int(vertices[j][0]))
                hlist.append(int(vertices[j][1]))
                print("Appended:", (int(vertices[j][0]), int(vertices[j][1]) ) )
            print("Initial vertices for a box completed.")
            # text: ymin, xmin, ymax, xmax
            # obj: ymin, xmin, ymax, xmax
            # order of parameters currently not synchronized with initial algorithm
            xmin, ymin, xmax, ymax = min(wlist), min(hlist), max(wlist), max(hlist)
            bbox.append((xmin, ymin, xmax, ymax))
            for j in range(4):
                p1 = (vertices[j][0], vertices[j][1])
                p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
                p1 = (int(p1[0]), int(p1[1]))
                #print("p1:",p1)
                p2 = (int(p2[0]), int(p2[1]))
                #print("p2:",p2)

                # Drawing line
                cv.line(frame, p1, p2, (0, 255, 0), 2, cv.LINE_AA)
                #cv.putText(frame, "{:.3f}".format(confidences[i[0]]), (vertices[0][0], vertices[0][1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)
                # Making rectangle and then applying it as a mask
                cv.rectangle(mask, (min(wlist), min(hlist)), (max(wlist), max(hlist)), 255, -1)
                masked = cv.bitwise_and(frame2, frame2, mask=mask)
        # Put efficiency information
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        
        # Test
        print("-----Printing out inital bbox-----:")
        for b in bbox:
            print(b)

        need_to_merge = True
        #Merge the boxes
        while need_to_merge:
            need_to_merge, bbox = merge_algo(bbox)

        print("-----Printing out final bbox-----:")
        counter = 0
        for b in bbox:
            counter += 1
            print(counter, b)
            # text: xmin, ymin, xmax, ymax
            # obj: xmin, ymin, xmax, ymax
            cv.rectangle(mask2, (b[0], b[1]), (b[2], b[3]), 255, -1)
            
            mask3 = np.zeros(frame2.shape[:2], dtype="uint8")
            cv.rectangle(mask3, (b[0], b[1]), (b[2], b[3]), 255, -1)
            masked3 = cv.bitwise_and(frame2, frame2, mask=mask3)

            # Likely would need to modify this line below if using Linux. Use this line to help with debugging. Would need to create box_images directory first.
            path = ".\\box_images\\box"+str(counter)+".jpg"
            cv.imwrite(path, masked3)

            #The coordinates for the cropping box are (left, upper, right, lower)
            image_to_be_cropped = Image.open(path)
            box = (b[0], b[1], b[2], b[3])
            cropped_image = image_to_be_cropped.crop(box)
            path2 = ".\\box_images\\box"+str(counter)+"_"+"cropped"+".jpg"
            cropped_image.save(path2)

            #saving variations of frames in rotations
            counter2 = 0
            rotatelist = [10,9,8,7,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10]
            masked3_copy = Image.open(path2)
            masked3_rotated = None
            for degree in rotatelist:
                counter2 += 1
                masked3_rotated = masked3_copy.rotate(degree)
                path3 = ".\\box_images\\box"+str(counter)+"_"+str(counter2)+".jpg"
                masked3_rotated.save(path3)
                print(path3, ":", pytesseract.image_to_string(masked3_rotated))


            

            
        # text: xmin, ymin, xmax, ymax
        # obj: xmin, ymin, xmax, ymax
        #merging frame2 and mask2 to make masked2 altered frame
        masked2 = cv.bitwise_and(frame2, frame2, mask=mask2)

        # Display the frame
        cv.imshow(kWinName,frame)
        cv.imwrite("output.png", frame)
        cv.imwrite("Rec.jpg", masked)
        cv.imwrite("Rec2.jpg", masked2)

        # applying OSD per individual box and printing out text after corrected rotation
        #im = Image.open("Rec2.jpg")
        #osd = pytesseract.image_to_osd(im, output_type="dict")
        #rotate = osd['rotate']
        #im_fixed = im.copy().rotate(rotate)

        #display(im_fixed.resize(int(0.3*s) for s in im_fixed.size)) #comment for now, since does not work
        #print(pytesseract.image_to_string(im_fixed))

        
        



