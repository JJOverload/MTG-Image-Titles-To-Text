def replace_bad_characters(original_string):
    stringbuffer = original_string
    index = 0

    for character in stringbuffer[:]:
        if character == '꞉': #You might say this is a normal semi-colon, but I say it's not.
            stringbuffer = stringbuffer[:index] + ":" + stringbuffer[index+1:]
        index = index + 1

    return stringbuffer


