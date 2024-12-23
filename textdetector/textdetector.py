# Import for text detection
import cv2 as cv
import math
import argparse
# Import for rectangle
import numpy as np
# Import for rotations (and pytesseract)
from PIL import Image
import pytesseract
#for autocorrect
import pandas as pd
import numpy as np
import textdistance
import re
from collections import Counter
# For jsonparser
import json
# For Timer
import datetime

# Recording start time for timer
starttime = datetime.datetime.now()

# Grabbing arguments from command line when executing command
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

# Location/name of answer .txt file
parser.add_argument('--answername', type=str, default="", help="Location/name of answer .txt file")


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

def merge_algo(bboxes): #bboxes is a list of bounding boxes data
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

#Finding Similar Names for autocorrector portion
def mtg_autocorrect(input_word, V, name_freq_dict, probs):
    #input_word = input_word.lower()

    # qval in similarities needs to be 2, meaning input_word needs to be 2 characters or more.
    if len(input_word) == 1:
        input_word = input_word + " "
    elif len(input_word) == 0:
        input_word = input_word + "  "
    similarities = [1-(textdistance.Jaccard(qval=2).distance(v,input_word)) for v in name_freq_dict.keys()]
    df = pd.DataFrame.from_dict(probs, orient='index').reset_index()
    df = df.rename(columns={'index':'Name', 0:'Prob'})
    df['Similarity'] = similarities
    output = df.sort_values(['Similarity', 'Prob'], ascending=False).head(1) #.iat[0,0]
    #print("output:\n", output)
    #if output.iat[0,2] <= 0.1:
    #    return("")
    return(output)

#Converting txt file true list for accuracy checker.
#Input: Text file name or location
#Output: List
def convertTxtFileAnswerToList(aname):
    outputlist = []
    f = open(aname, "r")
    for x in f: #x is a line of the text file
        outputlist.append(x.strip())
        print("During conversion -> Added to outputlist:|" + str(x.strip()) + "|")
    f.close()
    return(outputlist)

#Finding out the accuracy of results compared to answer list
def runAccuracyChecker(bestNameListNameOnly, answerList):
    a = answerList
    b = bestNameListNameOnly
    correctcounter = 0
    incorrectcounter = 0

    for nameofb in b:
        if nameofb in a:
            correctcounter = correctcounter + 1
            print("nameofb: " + str(nameofb) + " (Correct)")
            a.remove(nameofb)
        else: #incorrect
            incorrectcounter = incorrectcounter + 1
            print("nameofb: " + str(nameofb) + " (NOT Correct)")

    print("---------------------------------------------")
    print("correctcounter: " + str(correctcounter))
    print("incorrectcounter: " + str(incorrectcounter))
    print("Accuracy of bestNameListNameOnly results: " + str(correctcounter/(correctcounter+incorrectcounter)))


if __name__ == "__main__":
    # Read and store arguments
    confThreshold = args.thr
    nmsThreshold = args.nms
    inpWidth = args.width
    inpHeight = args.height
    model = args.model
    answerfilename = args.answername

    # Creating buffer/container for bounding boxes' vertices
    bbox = []
    # Setting distance limit for bounding box merging
    dist_limit = 50

    # Load network
    net = cv.dnn.readNet(model)
    if args.device == "cpu":
        net.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
        print("Using CPU device")
    elif args.device == "gpu": #not tested
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        print("Using GPU device")


    # Create a new named window
    #kWinName = "EAST: An Efficient and Accurate Scene Text Detector"
    #cv.namedWindow(kWinName, cv.WINDOW_NORMAL)
    outputLayers = []
    outputLayers.append("feature_fusion/Conv_7/Sigmoid")
    outputLayers.append("feature_fusion/concat_3")

    # Open a video file or an image file or a camera stream
    cap = cv.VideoCapture(args.input if args.input else 0)

    print("-----Opening Atomic Cards JSON------")
    names = []
    # AtomicCards_data is a dictionary of dictionaries of...
    with open('AtomicCards.json', 'r', encoding="utf8") as AtomicCards_file:
        AtomicCards_data = json.load(AtomicCards_file)
    # creating list/set of names
    names = list(AtomicCards_data["data"].keys()) # First list of names
    non_names = [] # Second list of names
    for n in names:
        for index in range(0, len(AtomicCards_data["data"][n])):
            print("Looking at: ", n, "| 'Side' number:", index+1)
            #print("-Storing", AtomicCards_data["data"][n][index]["text"], "into non_names...")
            if "text" in AtomicCards_data["data"][n][index]:
                non_names.append(json.dumps(AtomicCards_data["data"][n][index]["text"]))
            if "type" in AtomicCards_data["data"][n][index]:
                non_names.append(json.dumps(AtomicCards_data["data"][n][index]["type"]))

    # Non-names are also "vocab" we are using. Counting them as 
    # "names" for simplicity.
    names = names + non_names
    V = set(names)
    
    
    #Counter of name frequency
    name_freq_dict = {}
    name_freq_dict = Counter(names)
    #Relative Frequency of names
    print("-----Populating dictionary of Name Frequencies-----")
    probs = {}
    Total = sum(name_freq_dict.values())
    for k in name_freq_dict.keys():
        probs[k] = name_freq_dict[k]/Total


    print("-----Starting 'While' Looping (of one usually)------")
    #while cv.waitKey(1) < 0:
    # Read frame
    hasFrame, frame = cap.read()
    #if not hasFrame:
        #cv.waitKey()
        #break

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
    

    print("-----Printing out inital bbox-----:")
    for b in bbox:
        print(b)

    need_to_merge = True
    #Merge the boxes
    while need_to_merge:
        need_to_merge, bbox = merge_algo(bbox)

    print("-----Printing out final bbox-----:")
    bestNameList = [] #for best name list and similarity values
    bestNameListNameOnly = []
    counter = 0
    for b in bbox:
        counter += 1
        print(counter, b)
        # text: xmin, ymin, xmax, ymax
        # obj: xmin, ymin, xmax, ymax
        #Making rectangles for mask2, which will be be used to ultimately generate "Rec2.jpg"
        cv.rectangle(mask2, (b[0], b[1]), (b[2], b[3]), 255, -1)
        
        #Creating mask3 to create box image in "box_images" directory
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
        rotatelist = [6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6]
        masked3_copy = Image.open(path2)
        masked3_rotated = None
        maxSimilarity = 0.0
        bestOutput = mtg_autocorrect("", V, name_freq_dict, probs)
        startcountdown = False
        countdowncounter = 0
        i = 0 #index for rotatelist
        while(i < len(rotatelist)):
            degree = rotatelist[i]
            counter2 += 1
            masked3_rotated = masked3_copy.rotate(degree)
            path3 = ".\\box_images\\box"+str(counter)+"_"+str(counter2)+".jpg"
            masked3_rotated.save(path3)
            imageToStrStr = pytesseract.image_to_string(masked3_rotated)
            imageToStrStr = imageToStrStr.strip() #removing leading and trailing newlines/whitespace
            autocorrectOutput = mtg_autocorrect(imageToStrStr, V, name_freq_dict, probs)
            tempSimilarity = autocorrectOutput.iat[0,2]
            if tempSimilarity > maxSimilarity:
                maxSimilarity = tempSimilarity
                bestOutput = autocorrectOutput
                countdowncounter = 0 #reset countdown counter if max similarity is updated
            if maxSimilarity >= 0.6:
                startcountdown = True
            if startcountdown == True:
                countdowncounter = countdowncounter + 1
            print("countdowncounter: " + str(countdowncounter))
            print(path3 + ": '" + str(imageToStrStr) + "'\n" + str(autocorrectOutput))
            if countdowncounter >= 4:
                print("~Skipping rest of rotations~")
                break
            i = i + 1
            print("---------------------------------------------")
        print("Best Name:\n" + str(bestOutput)) #output the "best" name extracted from among all the rotated images for this bounding box
        
        if (bestOutput.iat[0,0] in non_names) or (bestOutput.iat[0,2] <= 0.40):
            print("-----Likely Noise/Non-name - Skipped-----")
        else: #If name is exising or is not "noise" due to low similarity
            bestNameList.append((bestOutput.iat[0,0], bestOutput.iat[0,2]))
            print("---------------------------------------------")



        

    print("-----Outputing Best Name List-----")
    print(bestNameList)
    print("Length of bestNameList: " + str(len(bestNameList)))
    for n, s in bestNameList:
        print(n)
        bestNameListNameOnly.append(n)
    # text: xmin, ymin, xmax, ymax
    # obj: xmin, ymin, xmax, ymax
    #merging frame2 and mask2 to make masked2 altered frame
    masked2 = cv.bitwise_and(frame2, frame2, mask=mask2)
    # Name of window
    kWinName = "MTG-Image-Titles-To-Text"
    # Spawn window
    cv.namedWindow(kWinName, cv.WINDOW_NORMAL)
    # Display the frame
    cv.imshow(kWinName,frame)
    cv.imwrite("output.png", frame)
    cv.imwrite("Rec.jpg", masked)
    cv.imwrite("Rec2.jpg", masked2)

    #execute AccuracyChecker, if argument is present in run command
    if (answerfilename!=""):
        answerList = convertTxtFileAnswerToList(answerfilename)
        runAccuracyChecker(bestNameListNameOnly, answerList)
    
    #Recording endtime and outputing elapsed time
    endtime = datetime.datetime.now()
    elapsedtime = endtime - starttime
    print("Elapsed Time:", elapsedtime)



