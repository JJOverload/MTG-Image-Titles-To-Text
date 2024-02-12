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

**References**
Good link if one wants to rely on non-online/website link for OCR: https://www.tensorflow.org/lite/examples/optical_character_recognition/overview

C++/Python article that references EAST (An Efficient and Accruate Scene Text Detector) paper:
https://learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/

StackOverflow on how to install multiple packages with one command: https://stackoverflow.com/questions/9956741/how-to-install-multiple-python-packages-at-once-using-pip


