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


cap = cv.imread(args.input, cv.IMREAD_GRAYSCALE)
thresh, cap = cv.threshold(cap, 127, 255, cv.THRESH_BINARY) #thresh is a dummy value
cap = cv.merge((cap,cap,cap)) 

cv.imwrite("test-output.jpg", cap)
