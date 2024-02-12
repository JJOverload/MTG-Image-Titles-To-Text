# Good code example in:
# https://www.kaggle.com/code/gauravduttakiit/autocorrect-with-python
# cd Documents\GitHub\MTG-Image-Titles-To-Text\
# https://www.imagetotext.info/jpg-to-word

import pandas as pd
import numpy as np
import textdistance
import re
from collections import Counter

import json

names = []
'''
with open('izzet-buylist.txt', 'r') as f:
    file_data = f.read()
    file_data = file_data.lower()
    names = re.findall("\\w+", file_data)
'''
'''
with open('izzet-buylist.txt', 'r') as f:
    names = f.readlines()
'''
print("-----Opening Atomic Cards JSON------")
with open('AtomicCards.json', 'r', encoding="utf8") as AtomicCards_file:
    AtomicCards_data = json.load(AtomicCards_file)
#AtomicCards_data is a dictionary of dictionaries of...
names = list(AtomicCards_data["data"].keys())
#print(names[:100])

V = set(names)
print(f"-----The first fifteen names in the text are: \n{names[0:15]}-----") #sixteen or fifteen?
print(f"-----There are {len(V)} unique names in the vocabulary.-----")

#Counter of name frequency
print("-----Printing out counter of name frequency of 15 most common names.-----")
name_freq_dict = {}
name_freq_dict = Counter(names)
print(name_freq_dict.most_common()[0:15])

#Relative Frequency of names
print("-----Printing out 15 most common names and their relative frequency within the whole file.-----")
probs = {}
Total = sum(name_freq_dict.values())
for k in name_freq_dict.keys():
    probs[k] = name_freq_dict[k]/Total
for commonName in name_freq_dict.most_common()[0:15]:
    print(commonName, probs[commonName[0]])

#Finding Similar Names
def mtg_autocorrect(input_word):
    #input_word = input_word.lower()
    if input_word in V:
        return("Your word seems to be correct")
    else:
        if len(input_word) == 1:
            input_word = input_word + " "
        elif len(input_word == 0):
            input_word = input_word + "  "
        similarities = [1-(textdistance.Jaccard(qval=2).distance(v,input_word)) for v in name_freq_dict.keys()]
        df = pd.DataFrame.from_dict(probs, orient='index').reset_index()
        df = df.rename(columns={'index':'Name', 0:'Prob'})
        df['Similarity'] = similarities
        output = df.sort_values(['Similarity', 'Prob'], ascending=False).head(1)#.iat[0,0]
        if output.iat[0,2] <= 0.1:
            return("This input is noise.")
        return(output)

search_word = "2222"
print("-----Printing out results for \"" + search_word + "\"------")
print(mtg_autocorrect(search_word))