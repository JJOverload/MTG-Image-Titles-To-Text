# MTG-Image-Titles-To-Text

**Setting Up EAST (An Efficient and Accurate Scene Text Detector) Code**

More info on how to set up EAST found here in this link:<br>
https://learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/


------------------------------------


**Installing Tesseract**
If wanting to use Tesseract. Will need to install it.
~~~
For Windows (check "Windows" section):
https://tesseract-ocr.github.io/tessdoc/Installation.html
~~~
~~~
For Ubuntu:

- For Tesseract:
`sudo apt install tesseract-ocr`

- For the development tools (Tesseract):
`sudo apt install libtesseract-dev`



For cv2 module in Linux:
`pip install opencv-python`
~~~

-----------------------------------------------------

**Install Packages for Autocorrecter**
If you don't have it already:
`pip install wheel`

then:
`pip install pandas`

`pip install Pyarrow`

Would need to install the "textdistance" package as well:
`pip install textdistance`


------------------------------------

**How to Run the Script**

Step 1: Go to the "textdetector" directory found in the repository.

Example: `cd Documents\GitHub\MTG-Image-Titles-To-Text\textdetector`

Step 2: Using python to run the program. (Be sure to make sure each dimension is divisible by 32)

Example:<br>
`python textdetector.py --input CardPileSample1.jpg --width 3072 --height 4096`<br>
`python textdetector.py --input tegwyll-nonlands-Copy.jpg --width 3072 --height 2656`<br>
`python textdetector.py --input 1_python-ocr.jpg --width 800 --height 352`<br>
`python textdetector.py --input tegwyll-nonlands-Copy-censored.jpg --width 3072 --height 2656`<br>

Note: Try to ensure that the image's height is not too large relative to width, since certain dimensions can cause the image to be rotated sideways. (As seen when using: `python textdetector.py --input tegwyll-nonlands.jpg --width 3072 --height 4064`)

------------------------------------

**What the Script Does**

- First uses EAST text detections to help detect the words off of the image of cards via bounding boxes.
- Then would use merging of bounding box to get a box around each title/name.
- Slight rotations of merged images gets applied before using text recognition algorithm on it (Pytesseract).
- Compare strings found for each rotated image to existing names gathered from data (Extracted from mtgjson.com), and keep the "best"/ones with the most similarities to existing MTG card name. Kept names gets displayed at the end of the program.

Note: Did not use OSD since it could not detect rotations less than 90 degrees with it.

------------------------------------

**TODO in Consideration**
- allow "better"(?) noise detection/analysis.
- add additional logic to handle double-faced/adventure cards
- upgrade merging algorithm to handle "triple overlapping" bboxes

------------------------------------

**Notes for textdetector.py**
Initial code from: https://learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/
Using this for reference as well: https://learnopencv.com/deep-learning-with-opencvs-dnn-module-a-definitive-guide/
https://github.com/spmallick/learnopencv/blob/master/TextDetectionEAST/textDetection.py
https://pyimagesearch.com/2021/01/19/image-masking-with-opencv/

Inspired by this code for applying merging of bounding boxes algorithm:
https://stackoverflow.com/questions/55593506/merge-the-bounding-boxes-near-by-into-one

Sample CMD commands:
cd Documents\GitHub\MTG-Image-Titles-To-Text\textdetector
`python textdetector.py --input CardPileSample1.jpg --width 3072 --height 4096`<br>
`python textdetector.py --input tegwyll-nonlands.jpg --width 3072 --height 4064`<br>
`python textdetector.py --input tegwyll-nonlands-Copy.jpg --width 3072 --height 2656`<br>
`python textdetector.py --input 1_python-ocr.jpg --width 800 --height 352`<br>
`python textdetector.py --input tegwyll-nonlands-Copy-censored.jpg --width 3072 --height 2656`<br>

------------------------------------

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

Pytesseract | Orientation & Script Detection (OSD):
https://www.kaggle.com/code/dhorvay/pytesseract-orientation-script-detection-osd

Correcting Text Orientation with Tesseract and Python:
https://pyimagesearch.com/2022/01/31/correcting-text-orientation-with-tesseract-and-python/

How to rotate an image using Python?:
https://www.geeksforgeeks.org/how-to-rotate-an-image-using-python/

Image Processing in Python with Pillow (Cropping Section)
https://auth0.com/blog/image-processing-in-python-with-pillow/



