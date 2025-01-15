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
# For replacing characters
from module import helper

# Grabbing arguments from command line when executing command
parser = argparse.ArgumentParser(description='Use this script to run text detection deep learning networks using OpenCV.')
# Input argument
parser.add_argument('--input', help='Path to input image or video file.')
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
parser.add_argument('--answername', type=str, default="", help="Location/name of answer .txt file.")

# Indicator of whether or not we show text onto Rec2.jpg
parser.add_argument('--showtext', action='store_true', help="Indicator of whether or not we show text onto Rec2.jpg image.")

args = parser.parse_args()

def find_good_thresh(cap):
    rows, cols = cap.shape
    string = ""
    lowest = 500
    highest = -1
    for y in range(0, rows):
        for x in range(0, cols):
            if cap[y][x] > highest:
                highest = cap[y][x]
            if cap[y][x] < lowest:
                lowest = cap[y][x]

    print("Lowest: " + str(lowest))
    print("Highest: " + str(highest))
    print("(Lowest+Highest)/2: " + str(int((lowest+highest)/2)))

    return(int((lowest+highest)/2))

cap = cv.imread(args.input, cv.IMREAD_GRAYSCALE)

print(cap)
print("------------------------")
print(cap.shape)
print(find_good_thresh(cap))


#thresh, cap = cv.threshold(cap, find_good_thresh(cap), 255, cv.THRESH_BINARY) #thresh is a dummy value
thresh, cap = cv.threshold(cap, find_good_thresh(cap), 255, cv.THRESH_BINARY) #thresh is a dummy value
cap = cv.merge((cap,cap,cap)) 

#cv.imwrite("test-output.jpg", cap)


string = "Breakneck Rider // Neck Breaker"
string = "B.F.M. (Big Furry Monster)"
print(string.find(" // "))
print(string[:string.find(" // ")] + "|" + string[string.find(" // ")+4:])

print(string[:string.find(" // ")])
print(string[string.find(" // ")+4:])
'''
for x in ["Breakneck Rider // Neck Breakers", "Arsonist Goblin", "Brazen // Borrower"]:
    print("Breakneck Rider" in x)

'''
print(min(['a','b','c'] + ['d','e','f']))