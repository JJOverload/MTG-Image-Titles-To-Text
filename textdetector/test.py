#print("Ratonhnhaké:ton Test".encode("utf-8"))
#test = b'Ratonhnhak\xc3\xa9\xea\x9e\x89ton'
#print(test.decode("utf-8"))

test = '꞉' #You might say this is a normal semi-colon, but I say it's not.
test = test.encode("utf-8")
print("Encoding below:")
print(test)
print("Then after decoding:")
print(test.decode("utf-8"))

#You might say this is a normal semi-colon, and I say it is as well. 
test = ':'
test = test.encode("utf-8")
print("Encoding below:")
print(test)
print("Then after decoding:")
print(test.decode("utf-8"))


# https://www.w3schools.com/python/python_modules.asp
