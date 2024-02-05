# From/based on code located at this link: https://analyticsvidhya.com/blog/2021/11/autocorrect-feature-using-nlp-in-python/
# Also found same code in: https://www.kaggle.com/code/gauravduttakiit/autocorrect-with-python 

import pandas as pd
import numpy as np
import textdistance
import re
from collections import Counter
words = []
with open('auto.txt', 'r') as f:
    file_name_data = f.read()
    file_name_data=file_name_data.lower()
    words = re.findall('w+',file_name_data)
# This is our vocabulary
V = set(words)
print("Top ten words in the text are:{words[0:10]}")
print("Total Unique words are {len(V)}.")
#Output:
#Top ten words in the text are:['moby', 'dick', 'by', 'herman', 'melville', '1851', 'etymology', 'supplied', 'by', 'a']
#Total Unique words are 17140.