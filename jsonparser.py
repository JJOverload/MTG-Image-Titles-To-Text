# When using cmd:
# cd Documents\GitHub\MTG-Image-Titles-To-Text\

import json


with open('CardTypes.json', 'r', encoding="utf8") as CardTypes_file:
    CardTypes_data = json.load(CardTypes_file)
    #print(CardTypes_data)
#print(json.dumps(CardTypes_data, indent=4))
print(CardTypes_data["data"]["types"]["artifact"]["subTypes"])


#Excerpt from JSON file
"""
"data": {
        "types": {
            "artifact": {
                "subTypes": [
                    "Attraction",
                    "Blood",
                    "Clue",
                    "Contraption",
                    "Equipment",
                    ...
"""

'''
with open('AtomicCards.json', 'r', encoding="utf8") as AtomicCards_file:
    AtomicCards_data = json.load(AtomicCards_file)
    #print(AtomicCards_data)
print(json.dumps(AtomicCards_data, indent=4))
'''
'''
with open('AllPrintings.json', 'r', encoding="utf8") as AllCards_file:
    AllCards_data = json.load(AllCards_file)
    #print(AllCards_data)
print(json.dumps(AllCards_data, indent=4))
'''

