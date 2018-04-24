
from os import path
import pyaudio
import time
import os
from pocketsphinx.pocketsphinx import *
from sphinxbase.sphinxbase import *

from espeak import espeak
espeak.set_parameter(espeak.Parameter.Pitch, 60)
espeak.set_parameter(espeak.Parameter.Rate, 110)
espeak.set_parameter(espeak.Parameter.Range, 600)
espeak.synth("Hey Guys My name is Jerry")
time.sleep(2)

MODELDIR = "/home/uawsscu/PycharmProjects/Pass1/object_recognition_detection/model_LG"
DATADIR = "/home/uawsscu/PycharmProjects/Pass1/object_recognition_detection/dataLG"

config = Decoder.default_config()
config.set_string('-logfn', '/dev/null')
config.set_string('-hmm', path.join(MODELDIR, 'en-us/en-us'))
config.set_string('-lm', path.join(MODELDIR, 'en-us/en-us.lm.bin'))
config.set_string('-dict', path.join(MODELDIR, 'en-us/cmudict-en-us.dict'))
decoder = Decoder(config)

# Switch to JSGF grammar
jsgf = Jsgf(path.join(DATADIR, 'sentence.gram'))
rule = jsgf.get_rule('sentence.move') #>> public <move>
fsg = jsgf.build_fsg(rule, decoder.get_logmath(), 7.5)
fsg.writefile('sentence.fsg')

decoder.set_fsg("sentence", fsg)
decoder.set_search("sentence")

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
stream.start_stream()

in_speech_bf = False
decoder.start_utt()

STPindex = 0
STPname =""
JOB =True
JOB_HowTo_Open = True
while True:
    buf = stream.read(1024)
    if buf:
        decoder.process_raw(buf, False, False)
        if decoder.get_in_speech() != in_speech_bf:
            in_speech_bf = decoder.get_in_speech()
            if not in_speech_bf:
                decoder.end_utt()

                try:
                    strDecode = decoder.hyp().hypstr
                    if strDecode != '':
                        #print strDecode
                            # >>>>>>> END <<<<<<<<<<<<
                        if JOB == True and strDecode[-3:] == 'end' and strDecode[:9] == "this is a":
                            JOB = False
                            print "\n------------------------------------------"
                            print '\nStream decoding result:', strDecode
                            print "\n------------------------------------------"

                            JOB = True

                            # >>>>>>> ARM <<<<<<<<<<<<
                        elif JOB == True and strDecode[:14] == 'this is how to':
                            JOB = False
                            JOB_HowTo_Open = True
                            print "\n------------------------------------------"
                            print '\nStream decoding result:', strDecode

                        elif JOB_HowTo_Open == True and strDecode == 'next':
                            print 'Stream decoding result:', strDecode
                            STPindex += 1
                            print STPindex, " : ", STPname
                                # SAVE Action
                        elif JOB_HowTo_Open == True and strDecode == 'stop':
                            JOB = True
                            JOB_HowTo_Open = False
                            STPindex = 0
                            print "STOP.."
                            print "\n------------------------------------------"

                        # >>>>>>> JERRY <<<<<<<<<<<<
                        elif strDecode[:5] == 'jerry':
                            print "\n------------------------------------------"
                            print '\nStream decoding result:', strDecode
                            print "\n------------------------------------------"


                        # >>>>>>> PASS DO YOU KNOW~??? <<<<<<<<<<<<
                        elif JOB == True and strDecode[:11] == 'do you know':
                            JOB = False
                            print "\n------------------------------------------"
                            print '\nStream decoding result:', strDecode
                            print "\n------------------------------------------"



                except AttributeError:
                    pass
                decoder.start_utt()


    else:
        break
decoder.end_utt()
print('An Error occured :', decoder.hyp().hypstr)

print "OK"
