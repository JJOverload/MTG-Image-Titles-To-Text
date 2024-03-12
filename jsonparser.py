# When using cmd:
# cd Documents\GitHub\MTG-Image-Titles-To-Text\

import json

'''
with open('CardTypes.json', 'r', encoding="utf8") as CardTypes_file:
    CardTypes_data = json.load(CardTypes_file)
    #print(CardTypes_data)
#print(json.dumps(CardTypes_data, indent=4))
print(CardTypes_data["data"]["types"]["artifact"]["subTypes"])
'''



with open('AtomicCards.json', 'r', encoding="utf8") as AtomicCards_file:
    AtomicCards_data = json.load(AtomicCards_file)


#print(AtomicCards_data["data"]["Abzan Kin-Guard"])
#print(json.dumps(AtomicCards_data["data"]["Abzan Kin-Guard"], indent=4))
print(json.dumps(AtomicCards_data["data"]["Abzan Kin-Guard"][0]["text"], indent=4))

print("------")
x = AtomicCards_data["data"]["Binding Geist // Spectral Binding"]
for y in range(0, len(x)):
    #keysList = list(x.keys())
    #print(keysList[:])
    print(x[y]["text"])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    print("------")

keysList = list(AtomicCards_data["data"].keys())
#print(keysList[:100])

print("------------------------Starting First 100 Keys--------------------------------")

counter = 0

for key in keysList[:100]:
    #print(key, counter, json.dumps(AtomicCards_data["data"][key][0]["text"]) )
    #print(key, counter)
    for y in range(0, len(AtomicCards_data["data"][key])):
        print(key, json.dumps(AtomicCards_data["data"][key][y]["text"]), counter)
        print("------")
    counter = counter + 1

print("-------------------------Starting Next 100 Keys-------------------------------")
for key in keysList[100:200]:
    #print(key, counter)
    for y in range(0, len(AtomicCards_data["data"][key])):
        print(key, json.dumps(AtomicCards_data["data"][key][y]["text"]), counter)
        print("------")
    counter = counter + 1


'''
with open('AllPrintings.json', 'r', encoding="utf8") as AllCards_file:
    AllCards_data = json.load(AllCards_file)
    #print(AllCards_data)
print(json.dumps(AllCards_data, indent=4))
'''

