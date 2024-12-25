# TCG MultiScan

A personal project for a TCG card title scanner. The goal is to have a tool that can recognize multiple cards in one go, in order to compile an accurate decklist of cards shown in a picture. Ideal scenario is to be compatible with pictures taken from modern phones nowadays.

------------------------------------

**What the "textdetector" Script Does**

- First uses EAST (An Efficient and Accurate Scene Text Detector) text detections to help detect the words off of the image of cards via bounding boxes.
- Then would use merging of bounding box to get a box around each title/name.
- Slight rotations of merged images gets applied before using text recognition algorithm on it (Pytesseract).
- Compare strings found for each rotated image to existing names gathered from data (Extracted from mtgjson.com), and keep the "best" ones with the most similarities to existing MTG card name. Kept names gets displayed at the end of the program.
- Provides accuracy checking if the "--answername" option is used. Option requires name of file for answer. Answer file (example: a .txt file) should have the proper names listed, with a name per line.

Note: Did not use OSD since it could not handle rotations less than 90 degrees with it.

------------------------------------

**Installation Instructions**

More info on how to set up EAST (An Efficient and Accurate Scene Text Detector) found here in this link:<br>
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


------------------------------------

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

Sample CMD commands:

`python textdetector.py --input 1_python-ocr.jpg --width 800 --height 352`

`python textdetector.py --input CardPileSample1.jpg --width 3072 --height 4096`

`python textdetector.py --input CardPileSample1.jpg --width 3072 --height 4096 --answername CardPileSample1-list-answer.txt`

`python textdetector.py --input CardPileSample1.jpg --width 3072 --height 4096 --answername CardPileSample1-list-answer.txt > output_sample.txt`

`python textdetector.py --input tegwyll-nonlands.jpg --width 3072 --height 4064`

`python textdetector.py --input tegwyll-nonlands-Copy.jpg --width 3072 --height 2656`

`python textdetector.py --input tegwyll-nonlands-Copy-censored.jpg --width 3072 --height 2656`

`python textdetector.py --input tegwyll-nonlands-Copy-censored.jpg --width 3072 --height 2656 --answername tegwyll-nonland-decklist-answer.txt`

`python textdetector.py --input tegwyll-nonlands-Copy-censored.jpg --width 3072 --height 2656 --answername tegwyll-nonland-decklist-answer.txt > output_tegwyll.txt`


NOTE: Try to ensure that the image's height is not too large relative to width, since certain dimensions can cause the image to be rotated sideways. (As seen when using: `python textdetector.py --input tegwyll-nonlands.jpg --width 3072 --height 4064`)

NOTE: textdetector code solely tested in Windows so far. If having issues with running in Linux, try changing the "path" variable assignment values (in the code) so that it uses forward slashes (/) instead of backslashes (\\) prevalent in Windows.

------------------------------------

**TODO in Consideration**
- add additional logic to handle double-faced/adventure cards
- Allow automatic refreshing of box image directory images (delete before populating)
- Test --answername option for full path variation
- Focus on other TCG card names

------------------------------------

**Notes for textdetector.py**

Initial code from: https://learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/

Using this for reference as well: 

https://learnopencv.com/deep-learning-with-opencvs-dnn-module-a-definitive-guide/

https://github.com/spmallick/learnopencv/blob/master/TextDetectionEAST/textDetection.py

https://pyimagesearch.com/2021/01/19/image-masking-with-opencv/

Inspired by this code for applying merging of bounding boxes algorithm:
https://stackoverflow.com/questions/55593506/merge-the-bounding-boxes-near-by-into-one


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



