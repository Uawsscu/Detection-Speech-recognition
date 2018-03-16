from textblob import TextBlob
#from capture import *
#from Kinect_detect import *

print "ss"
#>>>>>>>>>>>>>>>>>> PASS ThisIs! <<<<<<<<<<<<<<<<<<
def get_object_train(text):
    # CUT "END"
    nounP = ''
    np = False
    print "Train ---Obj---"
    ans = text[0:-3]
    #print "!!" + ans
    b = TextBlob(ans)
    for item in b.noun_phrases:
        #print item
        np = True
        nounP =item

    if(np==False) :
        sentence = b.sentences[0]
        #print sentence

        for word, pos in sentence.tags:
            if pos[0:1] == 'N':
                # CAPTURE
                #cap_ture(word)
                #print word + " >>N"
                nounP =word
                break
    return nounP

#>>>>>>>>>>>>>>>> PASS COMMAND! <<<<<<<<<<<<<<<<<<<<
def get_object_command(text):
    print "Command ---Obj---" #Jerry move a ball to the left #"Jerry grab a ball"
    ans = text[6:]

    return ans

#>>>>>>>>>>>>>>>>>> PASS TrainArm! <<<<<<<<<<<<<<<<
def text_STEP(text):
    text = text.split()
    STEP = ""
    for i in range(-2,0):
        STEP = STEP + " " + text[i]
        #print STEP+".."
    return STEP

def get_TrainArm(text):
    STEP = text_STEP(text) #"this is how to grab a ball one step" >>> grab a ball
    test = text.split(STEP,1)[0]
    test = test[15:]
    return test

#>>>>>>>>>>>>>>>>>> PASS Q! <<<<<<<<<<<<<<<<<<
def get_object_question(text):
    print "question--Obj--"
    # CUT "END"
    nounP = ''
    np = False
    b = TextBlob(text)
    for item in b.noun_phrases:
        np = True
        nounP = item

    if (np == False):
        sentence = b.sentences[0]
        for word, pos in sentence.tags:
            if pos[0:1] == 'N':
                nounP = word
                break
    return nounP
#>>>>>>>>>>> Enum <<<<<<<<<<<<<<<<<<<
def text2int(textnum, numwords={}):
    if not numwords:
        units = [
            "zero", "one", "two", "three", "four", "five"
        ]
        for idx, word in enumerate(units):    numwords[word] = (1, idx)

    current = result = 0
    for word in textnum.split():
        if word not in numwords:
            continue
            raise Exception("Illegal word: " + word)

        scale, increment = numwords[word]
        return increment

def get_objectDetect(text):
    np = False
    b = TextBlob(text)
    nounP=""
    for item in b.noun_phrases:
        np = True
        nounP =item

    if(np==False) :
        sentence = b.sentences[0]
        for word, pos in sentence.tags:
            if pos[0:1] == 'N':
                nounP =word
                break
    return nounP

#print get_object_question("do you know a red bottle")
#print get_object_command("Jerry grab a ball")
#print get_TrainArm("this is how to grab a ball one step")
#print get_object_train("This is a teddy bear end")
#print text2int("test fie is")
#print get_objectDetect("teddy bear")
