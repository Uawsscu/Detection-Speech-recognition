
import argparse as ap
import cv2
import imutils 
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *

#image_path = "/home/uawsscu/PycharmProjects/sift3/B-O-W/dataset/test/dall/dall7.jpg"


import numpy as np
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


# intializing the web camera device


#################################  DETECTIONS && PREDICTION ########################################


def detectBOW():
    clf, classes_names, stdSlr, k, voc = joblib.load("train.pkl")
    print "Ready!! Yessss"
    cap = cv2.VideoCapture(1)
    x=y=xh=yh=1
    font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
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
                vis_util.visualize_boxes_and_labels_on_image_array(image_np,
                                                                   np.squeeze(boxes),
                                                                   np.squeeze(classes).astype(np.int32),
                                                                   np.squeeze(scores),
                                                                   category_index,
                                                                   use_normalized_coordinates=True,
                                                                   line_thickness=8)

                ####_______________________________________######

                fea_det = cv2.FeatureDetector_create("SIFT")
                des_ext = cv2.DescriptorExtractor_create("SIFT")

                des_list = []
                try :
                    y = int(vis_util.f.getYmin() * 479.000)
                    yh = int(vis_util.f.getYmax() * 479.000)
                    x = int(vis_util.f.getXmin() * 639.000)
                    xh = int(vis_util.f.getXmax() * 639.000)
                    im = image_np[y:yh, x:xh]
                    kpts = fea_det.detect(im)
                    kpts, des = des_ext.compute(im, kpts)
                    des_list.append((im, des))

                    descriptors = des_list[0][1]
                    for image2, descriptor in des_list[0:]:
                        descriptors = np.vstack((descriptors, descriptor))

                    test_features = np.zeros((1, k), "float32")
                    for i in xrange(1):
                        words, distance = vq(des_list[i][1], voc)
                        for w in words:
                            test_features[i][w] += 1

                    nbr_occurences = np.sum((test_features > 0) * 1, axis=0)
                    idf = np.array(np.log((1.0 * 1 + 1) / (1.0 * nbr_occurences + 1)), 'float32')


                    test_features = stdSlr.transform(test_features)

                    #print clf.predict(test_features) >>>>>> class n. [0] [1]...[n]
                    predictions = [classes_names[i] for i in clf.predict(test_features)]

                    cv2.putText(image_np, str(predictions[0]), (x, y), font, 1,(0, 255, 255), 2)
                    #print predictions[0]

                except :
                    print('..')


                    #cv2.putText(image, prediction, pt, cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, [0, 255, 0], 2)

                cv2.imshow('image', cv2.resize(image_np, (640, 480)))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break



#################################### CAP IMAGE ###############################################


def capture():
    import time
    start = time.time()
    time.clock()
    elapsed = 0
    seconds = 200  # 20 S.
    cap = cv2.VideoCapture(1)
    # Running the tensorflow session
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
          elapsed = int(time.time() - start)
          print "EP : ", elapsed

          # Visualization of the results of a detection.
          vis_util.visualize_boxes_and_labels_on_image_array(image_np,
                                                             np.squeeze(boxes),
                                                             np.squeeze(classes).astype(np.int32),
                                                             np.squeeze(scores),
                                                             category_index,
                                                             use_normalized_coordinates=True,
                                                             line_thickness=8)

          cv2.imshow('image', cv2.resize(image_np, (640, 480)))

          if (elapsed % 10 == 0):

             try :
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

                 cv2.imwrite(
                     "/home/uawsscu/PycharmProjects/Pass1/object_recognition_detection/dataset/train/gohair" + str(
                         elapsed/10) + ".jpg",
                     crop_img, params)
                 print "OK cap"
                 cv2.destroyAllWindows()
             except :
                 print "no image PASS"

          if (elapsed >= seconds):
              break

          if cv2.waitKey(25) & 0xFF == ord('q'):
              cv2.destroyAllWindows()
              break









#capture()
detectBOW()

