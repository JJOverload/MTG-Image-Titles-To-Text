# Good code example in:
# https://www.kaggle.com/code/gauravduttakiit/autocorrect-with-python
# 
# https://www.imagetotext.info/jpg-to-word
#
# cd Documents\GitHub\MTG-Image-Titles-To-Text\

#for autocorrect
import pandas as pd
import numpy as np
import textdistance
import re
from collections import Counter
#for jsonparser
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
print(names[:100])


non_names = []
for n in names: #TESTING FOR NOW, need to remove slicing later
    for index in range(0, len(AtomicCards_data["data"][n])):
        print("-Looking at: ", n, index)
        #print("-Storing", AtomicCards_data["data"][n][index]["text"], "into non_names...")
        if "text" in AtomicCards_data["data"][n][index]:
            non_names.append(json.dumps(AtomicCards_data["data"][n][index]["text"]))
        if "type" in AtomicCards_data["data"][n][index]:
            non_names.append(json.dumps(AtomicCards_data["data"][n][index]["type"]))

#print(names[:100])
names = names + non_names


testnameofcard = "Aegis Turtle"
print("-----Printing out info with test statement using one-liner------")
print(json.dumps(AtomicCards_data["data"][testnameofcard], indent=4))

if ("text" in AtomicCards_data["data"][testnameofcard][0]):    
    print("-----Printing out info with test statement for text only------")
    x = AtomicCards_data["data"][testnameofcard][0]["text"]
    # convert into JSON and print the result as a JSON string:
    y = json.dumps(x, indent=4)
    print(y)
else:
    print("-----Info with test statement for text only DNE------")

if ("type" in AtomicCards_data["data"][testnameofcard][0]):
    print("-----Printing out info with test statement for text only 2------")
    x = AtomicCards_data["data"][testnameofcard][0]["type"]
    # convert into JSON and print the result as a JSON string:
    y = json.dumps(x, indent=4)
    print(y)
else:
    print("-----Info with test statement for text only 2 DNE------")






#------------------------------------------
'''
print("-----Opening Card Types JSON------")
with open('CardTypes.json', 'r', encoding='utf8') as CardTypes_file:
    CardTypes_data = json.load(CardTypes_file)
#CardTypes_data is the json data containing all of the information in the json file
#print("---------------Printing out cardtypes data---------------")
#print(CardTypes_data)

types = list(CardTypes_data["data"].keys())
'''

'''
if "vanguardd" in names:
    print("----------TRUE-------------")
else:
    print("----------FALSE-------------")
'''

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
    print(commonName)
    print(probs[commonName[0]])

#Finding Similar Names
def mtg_autocorrect(input_word):
    #input_word = input_word.lower()
    if input_word in V:
        return("Your word seems to be correct")
    else:
        # qval in similarities needs to be 2, meaning input_word needs to be 2 characters or more.
        if len(input_word) == 1:
            input_word = input_word + " "
        elif len(input_word) == 0:
            input_word = input_word + "  "
        similarities = [1-(textdistance.Jaccard(qval=2).distance(v,input_word)) for v in name_freq_dict.keys()]
        df = pd.DataFrame.from_dict(probs, orient='index').reset_index()
        df = df.rename(columns={'index':'Name', 0:'Prob'})
        df['Similarity'] = similarities
        output = df.sort_values(['Similarity', 'Prob'], ascending=False).head(1)#.iat[0,0]
        print("output:\n", output)
        print(output.iat[0,2])
        if output.iat[0,2] <= 0.1:
            return("")
        return(output)


unclean_lines = []
clean_lines = []
output = ""
with open('SampleWebsiteOCR.txt', 'r') as f:
    unclean_lines = f.readlines()
print("-----Printing out \"Unclean Lines\"-----")
print(unclean_lines)

for search_word in unclean_lines:
    search_word = search_word.strip() #removes all newline characters from beginning and ending, but not middle
    print("-----Printing out results for \"" + search_word + "\"------")
    output = mtg_autocorrect(search_word)
    if output == "":
        print("Skipped", search_word)
    else: #Not skipped, so should be "good"
        print("Adding", search_word)
        clean_lines.append(search_word)

print("-----Printing out \"clean Lines\"-----")
print(clean_lines)

with open('output-from-autocorrector.txt', 'w') as f:
    for i in range(0, len(clean_lines)):
        f.write("1 " + clean_lines[i] + "\n")
