#code found in: https://builtin.com/data-science/python-ocr

from PIL import Image
import pytesseract
import numpy as np

#For image preprocessing of second noisy image
#import numpy as np
import cv2

filename = "1_python-ocr.jpg"
img1 = np.array(Image.open(filename))
text = pytesseract.image_to_string(img1)

print("First image text:", text)

filename = '3_python-ocr.jpg'
img2 = np.array(Image.open(filename))
text = pytesseract.image_to_string(img2)
print("Second image text:", text)

#-------------------------------

#Cleaning up noisy image
#norm_img = np.zeros((
