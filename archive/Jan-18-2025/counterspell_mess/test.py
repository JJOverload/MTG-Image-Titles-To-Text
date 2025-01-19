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
def convertMTGJSONNamesToVocabAndNames(jsonName):
    names = []
    non_names = [] # Second list of names reserved for things like type info and text of cards 
    unaltered_double_names = []
    separated_double_names = []
    double_temp_list = []
    V = {}
    # AtomicCards_data is a dictionary of dictionaries of MTG card data...
    with open(jsonName, 'r', encoding='utf-8') as AtomicCards_file:
        AtomicCards_data = json.load(AtomicCards_file)
    # creating list/set of names
    names = list(AtomicCards_data["data"].keys()) # First list of names
    #for each "name" in names
    for n in names:
        if len(AtomicCards_data["data"][n]) > 1:
            print("Initially looking at: ", helper.replace_bad_characters(n), "| 'Side' total:", len(AtomicCards_data["data"][n]))
            unaltered_double_names.append(n)
            double_temp_list = split_double_card_names(n)
            for dn in double_temp_list:
                separated_double_names.append(dn)
                print("Appending dn to double_names:", helper.replace_bad_characters(dn))
        for index in range(0, len(AtomicCards_data["data"][n])):
            #print("Looking at: ", n, "| 'Side' number:", index+1)
            print("Looking at: ", helper.replace_bad_characters(n), "| 'Side' number:", index+1)
            #print("-Storing", AtomicCards_data["data"][n][index]["text"], "into non_names...")
            if "text" in AtomicCards_data["data"][n][index]:
                non_names.append(json.dumps(AtomicCards_data["data"][n][index]["text"]))
            if "type" in AtomicCards_data["data"][n][index]:
                non_names.append(json.dumps(AtomicCards_data["data"][n][index]["type"]))

    # Non-names are also "vocab" we are using. Counting them as "names" for simplicity.
    #print()
    #print(separated_double_names)
    #print()
    for udn in unaltered_double_names:
        names.remove(udn)
    names = separated_double_names + names + non_names
    V = set(names) # V for vocab

    #for vocab in V:
    #    if vocab == "Brazen Borrower":
    #        print("FOUND IT ---------------------------<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    return(V, names, non_names, separated_double_names, unaltered_double_names)

def split_double_card_names(n):
    if n.find(" // ") != -1:
        return([n[:n.find(" // ")], n[n.find(" // ")+4:]])
    return([n])

def mtg_autocorrect(input_word, V, name_freq_dict, probs):
    #input_word = input_word.lower()

    # qval in similarities needs to be 2, meaning input_word needs to be 2 characters or more.
    if len(input_word) == 2:
        input_word = input_word + " "
    if len(input_word) == 1:
        input_word = input_word + "  "
    elif len(input_word) == 0:
        input_word = input_word + "   "
    similarities = [1-(textdistance.Jaccard(qval=3).distance(v,input_word)) for v in name_freq_dict.keys()]
    #similarities = [(textdistance.Sorensen(qval=2).similarity(v,input_word)) for v in name_freq_dict.keys()]
    df = pd.DataFrame.from_dict(probs, orient='index').reset_index()
    df = df.rename(columns={'index':'Name', 0:'Prob'})
    df['Similarity'] = similarities
    output = df.sort_values(['Similarity', 'Prob'], ascending=False).head(1) #.iat[0,0]

    #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #tempoutput = df.sort_values(['Similarity', 'Prob'], ascending=False).head(20) #.iat[0,0]
    #print(tempoutput)
    #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
    #print("output:\n", output)
    #if output.iat[0,2] <= 0.1:
    #    return("")
    return(output)

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
    
    int_result = int((int(lowest)+int(highest))/2)
    print("Lowest: " + str(lowest))
    print("Highest: " + str(highest))
    print("(Lowest+Highest)/2: " + str(int_result))

    return(int_result)

V, names, non_names, separated_double_names, unaltered_double_names = convertMTGJSONNamesToVocabAndNames("..\\AtomicCards.json")

path = "box1_cropped.jpg"
#image = cv.imread(path, cv.IMREAD_GRAYSCALE)
image = cv.imread(path)
#thresh, masked3_BnW = cv.threshold(image, find_good_thresh(image), 255, cv.THRESH_BINARY_INV)
#masked3_BnW = cv.adaptiveThreshold(image,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,3,2)
#cv.imwrite("image0.jpg", masked3_BnW)
#masked3_BnW = (255 - masked3_BnW)
#cv.imwrite("image1.jpg", masked3_BnW)
#masked3_BnW = cv.merge((masked3_BnW, masked3_BnW, masked3_BnW))
#cv.imwrite("image2.jpg", masked3_BnW)
#kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
#sharpened_black_and_white = cv.filter2D(masked3_BnW, -1, kernel)
cv.imwrite("image3.jpg", image)

masked3_copy = Image.open("image3.jpg")
rotatelist = [6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6]

name_freq_dict = Counter(names)
probs = {}
Total = sum(name_freq_dict.values())
for k in name_freq_dict.keys():
    probs[k] = name_freq_dict[k]/Total

i = 0 #index for rotatelist
while(i < len(rotatelist)):
    degree = rotatelist[i]
    masked3_rotated = masked3_copy.rotate(degree)
    path3 = ".\\test"+".jpg"
    masked3_rotated.save(path3)
    imageToStrStr = pytesseract.image_to_string(masked3_rotated)
    imageToStrStr = imageToStrStr.strip() #removing leading and trailing newlines/whitespace
    print("Degree: " + str(degree))
    print("imageToStrStr: " + imageToStrStr)
    autocorrectOutput = mtg_autocorrect(imageToStrStr, V, name_freq_dict, probs)
    print(autocorrectOutput)
    i += 1