import time
start = time.time()
time.clock()
elapsed = 0
seconds = 60 # 20 S.

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
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#intializing the web camera device

import cv2



#################################### CAP IMAGE ###############################################


def capture(namePath,obj_name):
    print "CAPPP"
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
                vis_util.visualize_boxes_and_labels_on_image_array(image_np,
                                                                   np.squeeze(boxes),
                                                                   np.squeeze(classes).astype(np.int32),
                                                                   np.squeeze(scores),
                                                                   category_index,
                                                                   use_normalized_coordinates=True,
                                                                   line_thickness=8)

                cv2.imshow('image', cv2.resize(image_np, (640, 480)))

                if (elapsed % 10 == 0):

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

                        cv2.imwrite(namePath + obj_name + str(
                            elapsed / 10) + ".jpg",
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
def detectBOW():
    import time
    start = time.time()
    time.clock()
    elapsed = 0
    seconds = 20  # 20 S.

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
          if (elapsed >= seconds):
              objName = st.split("#")[0]
              print objName
              cv2.destroyAllWindows()
              break

          if cv2.waitKey(25) & 0xFF == ord('q'):
              cv2.destroyAllWindows()
              break

def detectBOW2():

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

          cv2.imshow('image', cv2.resize(image_np, (640, 480)))
          st = vis_util.f.getPredic()
          if st != ""  :
              st = st.split("#")
              objName = st[0]
              st2 = st[1].split(",")
              Xmax = st2[3]
              Xmin = st2[2]
              K=  (int(Xmax)+int(Xmin))/2
              st3 = objName + " " + str(K)
              print st3


          elapsed = int(time.time() - start)

          cv2.imshow('image', cv2.resize(image_np, (640, 480)))
         #if Not print "I understand"
          #if (elapsed >= seconds):
            #  return st3
            #  cv2.destroyAllWindows()

          if cv2.waitKey(25) & 0xFF == ord('q'):
              cv2.destroyAllWindows()
              break

print detectBOW2()