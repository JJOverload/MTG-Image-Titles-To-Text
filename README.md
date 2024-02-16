# MTG-Image-Titles-To-Text

**Install Packages**
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


**References**

Good link if one wants to rely on non-online/website link for OCR: https://www.tensorflow.org/lite/examples/optical_character_recognition/overview

C++/Python article that references EAST (An Efficient and Accruate Scene Text Detector) paper:
https://learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/

- Example code from the above link:
- https://github.com/spmallick/learnopencv/blob/master/TextDetectionEAST/textDetection.py

StackOverflow on how to install multiple packages with one command: https://stackoverflow.com/questions/9956741/how-to-install-multiple-python-packages-at-once-using-pip


A good read. Skimmed through the progress made so far by Quentin Fortier. Should be able to learn some stuff from here:
https://fortierq.github.io/mtgscan-ocr-azure-flask-celery-socketio/

Python OCR with Tesseract:
https://builtin.com/data-science/python-ocr

Optical Character Recognition Using TensorFlow:
https://medium.com/analytics-vidhya/optical-character-recognition-using-tensorflow-533061285dd3

Merge the Bounding boxes near by into one (StackOverflow):
https://stackoverflow.com/questions/55593506/merge-the-bounding-boxes-near-by-into-one

Tesseract Installation Guide:
https://tesseract-ocr.github.io/tessdoc/Installation.html

