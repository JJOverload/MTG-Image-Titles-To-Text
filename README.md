# MTG-Image-Titles-To-Text

**Install Packages for Autocorrecter**
If you don't have it already:
`pip install wheel`

then:
`pip install pandas`

`pip install Pyarrow`

Would need to install the "textdistance" package as well:
`pip install textdistance`

------------------------------------





**The Process Plan So Far**

Use this link: https://www.imagetotext.info/jpg-to-word

Then:
- Save output from link into a text file
- (Might need to remove noise somehow)
- Run autocorrector.py to have corrections in case missing minor spelling.

------------------------------------

**Tesseract Route**
If wanting to pivot with Tesseract. Will need to install it.

For Ubuntu:

- For Tesseract:
`sudo apt install tesseract-ocr`

- For the development tools (Tesseract):
`sudo apt install libtesseract-dev`



For cv2 module in Linux:
`pip install opencv-python`

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For Windows (check "Windows" section):
https://tesseract-ocr.github.io/tessdoc/Installation.html
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
-----------------------------------------------------


Note: Multiple rotations is needed for each pile, since it is usually not possible to have every card to be in the same orientation. Might need to do additional pivot and allow card by card scanning.






Idea: Use EAST detector to locate locations of text, isolate the surrounding area (using masking) of the detected box, then use Tesseract's OSD (Orientation and script detection) to get the proper orientation and then recognize the text. Process image before or during as needed.
- Source: https://pyimagesearch.com/2022/01/31/correcting-text-orientation-with-tesseract-and-python/








**References**

Good link if one wants to rely on non-online/website link for OCR: https://www.tensorflow.org/lite/examples/optical_character_recognition/overview

C++/Python article that references EAST (An Efficient and Accruate Scene Text Detector) paper:
https://learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/

- Example code from the above link:
- https://github.com/spmallick/learnopencv/blob/master/TextDetectionEAST/textDetection.py

StackOverflow on how to install multiple packages with one command: https://stackoverflow.com/questions/9956741/how-to-install-multiple-python-packages-at-once-using-pip


(Paper) A good read. Skimmed through the progress made so far by Quentin Fortier. Should be able to learn some stuff from here:
https://fortierq.github.io/mtgscan-ocr-azure-flask-celery-socketio/


Optical Character Recognition Using TensorFlow:
https://medium.com/analytics-vidhya/optical-character-recognition-using-tensorflow-533061285dd3

Merge the Bounding boxes near by into one (StackOverflow):
https://stackoverflow.com/questions/55593506/merge-the-bounding-boxes-near-by-into-one

Tesseract Installation Guide:
https://tesseract-ocr.github.io/tessdoc/Installation.html

Python (Tesseract) OCR Installation:
https://builtin.com/data-science/python-ocr

Image Masking with OpenCV:
https://pyimagesearch.com/2021/01/19/image-masking-with-opencv/



