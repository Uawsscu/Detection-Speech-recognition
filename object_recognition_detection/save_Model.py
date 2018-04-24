############################################# IMPORT Tensorflow #########################################################

import os
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#-----------------Model-----------------

import cv2
import imutils
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler



####################################################################################################
#################################  DETECTIONS && PREDICTION ########################################

def detectBOW2():
    import time
    start = time.time()
    time.clock()
    elapsed = 0
    seconds = 25  # 20 S.
    cap = cv2.VideoCapture(1)
    vis_util.f.setPredic("")

    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
       ret = True
       while (ret):
          ret,image_np = cap.read()
          image_np_expanded = np.expand_dims(image_np, axis=0)
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          vis_util.visualize_boxes_and_labels_on_image_array(image_np,
                                                             np.squeeze(boxes),
                                                             np.squeeze(classes).astype(np.int32),
                                                             np.squeeze(scores),
                                                             category_index,
                                                             use_normalized_coordinates=True,
                                                             line_thickness=8)
          elapsed = int(time.time() - start)
          cv2.imshow('image', cv2.resize(image_np, (640, 480)))
          st = vis_util.f.getPredic()
          objName=""
          if st != ""  :
              st = st.split("#")
              objName = st[0]
              st2 = st[1].split(",")
              Xmax = st2[3]
              Xmin = st2[2]
              K=  (int(Xmax)+int(Xmin))/2
              st3 = objName + " " + str(K)
              #print st3
          if (elapsed >= seconds):
              with sqlite3.connect("Test_PJ2.db") as con:
                  cur=con.cursor()
                  try:
                      cur.execute("UPDATE call_Detect SET Name=?,K=? WHERE ID = 1", (objName, K))
                  except :
                      print "I can not see (not update) "
              break
              cv2.destroyAllWindows()

          if cv2.waitKey(25) & 0xFF == ord('q'):
              cv2.destroyAllWindows()
              break

    cap.release()
    cv2.destroyAllWindows()



#################################### CAP IMAGE ###############################################


def capture(namePath,obj_name,count):
    print "CAPPP"
    import time
    start = time.time()
    time.clock()
    elapsed = 0
    i=0
    seconds = 200  # 20 S.
    cap = cv2.VideoCapture(1)
    # Running the tensorflow session
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            ret = True
            while (ret):
                ret, image_np = cap.read()
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                elapsed = int(time.time() - start)
                print "EP : ", elapsed

                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array2(image_np,
                                                                   np.squeeze(boxes),
                                                                   np.squeeze(classes).astype(np.int32),
                                                                   np.squeeze(scores),
                                                                   category_index,
                                                                   use_normalized_coordinates=True,
                                                                   line_thickness=8)

                cv2.imshow('image', cv2.resize(image_np, (640, 480)))

                if (elapsed % 10 == 0):
                    i+=1

                    try:
                        y = int(vis_util.f.getYmin() * 479.000)
                        yh = int(vis_util.f.getYmax() * 479.000)
                        x = int(vis_util.f.getXmin() * 639.000)
                        xh = int(vis_util.f.getXmax() * 639.000)
                        print y, " ", yh, " ", x, " ", xh
                        cv2.imshow('RGB image', image_np)

                        params = list()
                        # 143 : 869 // 354 :588
                        # 120:420, 213:456
                        crop_img = image_np[y:yh, x:xh]

                        cv2.imwrite(namePath + obj_name + str(count) + "_" + str(i) + ".jpg",
                                    crop_img, params)
                        i+=1
                        cv2.imwrite(namePath + obj_name + str(count) + "_" + str(i) + ".jpg",
                                    crop_img, params)
                        print "OK cap"
                        cv2.destroyAllWindows()
                    except:
                        print "no image PASS"

                if (elapsed >= seconds):
                    cv2.destroyAllWindows()
                    break

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
    cap.release()
    cv2.destroyAllWindows()


########################################################################################################
########################################## SAVE MODEL ##################################################

def save_model():
    train_path = "/home/uawsscu/PycharmProjects/Pass1/object_recognition_detection/pic"


    training_names = os.listdir(train_path)

    image_paths = []
    image_classes = []  ## 00000,111111,2222,33333
    class_id = 0
    for training_name in training_names:
        dir = os.path.join(train_path, training_name)
       # print("dir : ",dir)
        class_path = imutils.imlist(dir)
       # print class_path," classPath Type : ",type(class_path)
        image_paths += class_path
        image_classes += [class_id] * len(class_path)
      #  print " image class : ",image_classes
        class_id += 1

    # Create feature extraction and keypoint detector objects
    # print image_classes," imP :",image_paths

    fea_det = cv2.FeatureDetector_create("SIFT")
    des_ext = cv2.DescriptorExtractor_create("SIFT")

    # List where all the descriptors are stored
    des_list = []

    for image_path in image_paths:

        im = cv2.imread(image_path)
        kpts = fea_det.detect(im)
       # print "kpt : ",kpts
        kpts, des = des_ext.compute(im, kpts)
        des_list.append((image_path, des))
        #print image_path
        #print "des : ",des

        # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]
   # print "dess : ",descriptors

    for image_path, descriptor in des_list[1:]:
        try :
           # print "des2 : ",descriptor
            descriptors = np.vstack((descriptors, descriptor))

        except :
           # print image_paths
            pass

        # Perform k-means clustering

    k = 7
    voc, variance = kmeans(descriptors, k, 1)

    #print voc

    # Calculate the histogram of features
    im_features = np.zeros((len(image_paths), k), "float32")  # len(ALL pic) >> [0000000][00000]...
    #print im_features #[00000][0000]


    for i in xrange(len(image_paths)):
      try :
          words, distance = vq(des_list[i][1], voc)
          for w in words:
              im_features[i][w] += 1
              print im_features
      except :
          print "pass 339"
          pass

    # Perform Tf-Idf vectorization

    nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
    idf = np.array(np.log((1.0 * len(image_paths) + 1) / (1.0 * nbr_occurences + 1)), 'float32')
    im_features = np.multiply(im_features, idf)

    # Scaling the words
    stdSlr = StandardScaler().fit(im_features)
    #print stdSlr
    im_features = stdSlr.transform(im_features)
    #print im_features
    clf = LinearSVC()
    clf.fit(im_features, np.array(image_classes))
    # Save the SVM

    joblib.dump((clf, training_names, stdSlr, k, voc), "train.pkl", compress=3)
    print clf," ", training_names ," ", stdSlr," ",k," ",voc
    print "SAVE Model"

#test2()
save_model()
obj_name = "teddy"
p = "/home/uawsscu/PycharmProjects/Pass1/object_recognition_detection/pic/" + obj_name + "/"
#capture(p, obj_name, 1)

#detectBOW2()