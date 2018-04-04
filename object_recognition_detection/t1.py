from manageDB import *
from main1 import *
import os


JOB = False
print "\n------------------------------------------"
obj_name = "ball" # sentence to word

print '\nStream decoding result:', obj_name

print "Speech : ", obj_name
                            # create folder
dataset_Path = r'/home/uawsscu/PycharmProjects/Pass1/object_recognition_detection/pic/' + obj_name
p="/home/uawsscu/PycharmProjects/Pass1/object_recognition_detection/pic/"+ obj_name+"/"

if not os.path.exists(dataset_Path):
    print "New Data"
    os.makedirs(dataset_Path)
    capture(p,obj_name,1)  #capture image for train >> SAVE IMAGE
    lenObj = int(lenDB("Corpus_Main.db", "SELECT * FROM obj_ALL2"))  # count ROWs
    insert_object_Train(obj_name, int(lenObj + 1))  # check Found objects?
else:
    count = int(search_object_Train2(obj_name))
    capture(p, obj_name,count+1)####cap2
    update_object_Train2(count+1,obj_name) #UPDATE COUNT++


JOB = True
save_model()
