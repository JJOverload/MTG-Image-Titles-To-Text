# Good code example in:
# https://www.kaggle.com/code/gauravduttakiit/autocorrect-with-python
# cd Documents\GitHub\MTG-Image-Titles-To-Text\

import pandas as pd
import numpy as np
import textdistance
import re
from collections import Counter

words = []
'''
with open('izzet-buylist.txt', 'r') as f:
    file_data = f.read()
    file_data = file_data.lower()
    words = re.findall("\\w+", file_data)
'''
with open('izzet-buylist.txt', 'r') as f:
    words = f.readlines()

V = set(words)
print(f"The first sixteen words in the text are: \n{words[0:15]}")
print(f"There are {len(V)} unique words in the vocabulary.")

#Counter of word frequency
print("-----Printing out counter of word frequency of 15 most common words.-----")
word_freq_dict = {}
word_freq_dict = Counter(words)
print(word_freq_dict.most_common()[0:15])

#Relative Frequency of words
print("-----Printing out 15 most common words and their relative frequency within the whole file.-----")
probs = {}
Total = sum(word_freq_dict.values())
for k in word_freq_dict.keys():
    probs[k] = word_freq_dict[k]/Total
for commonWord in word_freq_dict.most_common()[0:15]:
    print(commonWord, probs[commonWord[0]])

#Finding Similar Words
def mtg_autocorrect(input_word):
    input_word = input_word.lower()
    if input_word in V:
        return("Your word seems to be correct")
    else:
        similarities = [1-(textdistance.Jaccard(qval=2).distance(v,input_word)) for v in word_freq_dict.keys()]
        df = pd.DataFrame.from_dict(probs, orient='index').reset_index()
        df = df.rename(columns={'index':'Word', 0:'Prob'})
        df['Similarity'] = similarities
        output = df.sort_values(['Similarity', 'Prob'], ascending=False).head()
        return(output)

search_word = "arcane signet"
print("-----Printing out results for " + search_word + "------")
print(mtg_autocorrect(search_word))