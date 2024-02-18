#code found in: https://builtin.com/data-science/python-ocr
# cd MTG-Image-Titles-To-Text/tesseract-ocr

from PIL import Image
import pytesseract
import numpy as np

#For image preprocessing of second noisy image
#import numpy as np
import cv2

#for Windows - Might need to comment this out if in Linux
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

filename = "1_python-ocr.jpg"
img1 = np.array(Image.open(filename))
text = pytesseract.image_to_string(img1)

print("First image text:", text)

#filename = '3_python-ocr.jpg'
#filename = "CardPileSample1.jpg"
filename = "tegwyll-nonlands-Copy.jpg"
img2 = np.array(Image.open(filename))
text = pytesseract.image_to_string(img2)
print("Second image text:", text)

#-------------------------------

#Using the same filename variable above
img = np.array(Image.open(filename))
#Cleaning up noisy image
norm_img = np.zeros((img.shape[0], img.shape[1]))
img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]
img = cv2.GaussianBlur(img, (1, 1), 0)
cv2.imwrite("cleaned-output.jpg", img)
text = pytesseract.image_to_string(img)
print("Second image text (from cleaned):", text)




