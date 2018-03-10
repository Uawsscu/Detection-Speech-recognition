import cv2
import numpy as np
import time

def Detect(objectName):
    font = cv2.FONT_HERSHEY_SIMPLEX
    MIN_MATCH_COUNT = 25
    try :
        setpath = '/home/uawsscu/PycharmProjects/Pass1/object_recognition_detection/pic/' + objectName + '.jpg'
        detector = cv2.SIFT()

        FLANN_INDEX_KDITREE = 0
        flannParam = dict(algorithm=FLANN_INDEX_KDITREE, tree=5)
        flann = cv2.FlannBasedMatcher(flannParam, {})

        trainImg = cv2.imread(setpath, 0)
        trainKP, trainDesc = detector.detectAndCompute(trainImg, None)
        print trainKP
    except :
        print ".."


    QUESTION_COUNT = 0
    goodCount = 0
    cap = cv2.VideoCapture(1)
    ret = True
    while (ret):
        try :
            ret, QueryImgBGR = cap.read()


            QueryImg = cv2.cvtColor(QueryImgBGR, cv2.COLOR_BGR2GRAY)
            queryKP, queryDesc = detector.detectAndCompute(QueryImg, None)
            matches = flann.knnMatch(queryDesc, trainDesc, k=2)

            goodMatch = []
            for m, n in matches:
                if m.distance < 0.65 * n.distance:
                    goodMatch.append(m)
            if len(goodMatch) > MIN_MATCH_COUNT:

                print "OKKKKKK"

                tp = []
                qp = []
                for m in goodMatch:
                    tp.append(trainKP[m.trainIdx].pt)
                    qp.append(queryKP[m.queryIdx].pt)
                tp, qp = np.float32((tp, qp))
                H, status = cv2.findHomography(tp, qp, cv2.RANSAC, 3.0)
                h, w = trainImg.shape

                trainBorder = np.float32([[[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]])
                queryBorder = cv2.perspectiveTransform(trainBorder, H)
                arr_int = np.int32(queryBorder)
                cv2.polylines(QueryImgBGR, [arr_int], True, (0, 255, 0), 5)

                on_left = arr_int[0][0]
                #print arr_int[0][1]
                #print arr_int[0][2]
                #print arr_int[0][3]





                cv2.putText(QueryImgBGR, objectName, (arr_int[0][0][0] ,arr_int[0][0][1] ), font, 1.5, (255, 255, 255), 3)
                #cv2.putText(img, str(id), (x, y + h), font, 3, (255, 255, 255), 2, cv2.LINE_AA);

            else:
                print "Not Enough -->> %d : %d" % (len(goodMatch), MIN_MATCH_COUNT)

            QueryImgBGR = cv2.drawKeypoints(QueryImgBGR, queryKP)
            cv2.imshow('result', QueryImgBGR)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        except :
            print "Error"

Detect("herbal30")
#Detect("ball", 'command')
