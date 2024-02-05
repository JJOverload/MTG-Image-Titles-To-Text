# When using cmd:
# cd Documents\GitHub\MTG-Image-Titles-To-Text\

import json
'''
with open('AllPrintings.json', 'r', encoding="utf8") as AllCards_file:
    AllCards_data = json.load(AllCards_file)
    #print(AllCards_data)
print(json.dumps(AllCards_data, indent=4))
'''
'''
with open('CardTypes.json', 'r', encoding="utf8") as AtomicCards_file:
    AtomicCards_data = json.load(AtomicCards_file)
    #print(AtomicCards_data)
print(json.dumps(AtomicCards_data, indent=4))
'''

with open('AtomicCards.json', 'r', encoding="utf8") as AtomicCards_file:
    AtomicCards_data = json.load(AtomicCards_file)
    #print(AtomicCards_data)
print(json.dumps(AtomicCards_data, indent=4))

