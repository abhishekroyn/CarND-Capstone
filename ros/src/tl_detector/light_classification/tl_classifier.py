#!/usr/bin/env python
import rospy
import tf
from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
from imutils import *
import cv2

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        # pass
        self.model = tf.Graph()

        # create a context manager that makes this model the default one for
        # execution
        with self.model.as_default():
            # initialize the graph definition
            graphDef = tf.GraphDef()

            # load the graph from disk
            with tf.gfile.GFile("/home/student/Desktop/CarND-Capstone/ros/src/tl_detector/light_classification/graph_optimized.pb", "rb") as f:
                serializedGraph = f.read()
                graphDef.ParseFromString(serializedGraph)
                tf.import_graph_def(graphDef, name="")

            # create a session to perform inference
            self.sess = tf.Session(graph=self.model)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        model = self.model

        sess = self.sess

        #TODO implement light color prediction
        # grab a reference to the input image tensor and the boxes
        # tensor

        imageTensor = model.get_tensor_by_name("image_tensor:0")
        boxesTensor = model.get_tensor_by_name("detection_boxes:0")

        # for each bounding box we would like to know the score
        # (i.e., probability) and class label
        scoresTensor = model.get_tensor_by_name("detection_scores:0")
        classesTensor = model.get_tensor_by_name("detection_classes:0")
        numDetections = model.get_tensor_by_name("num_detections:0")

        # load the image from disk
        (H, W) = image.shape[:2]

        # check to see if we should resize along the width
        if W > H and W > 1000:
            image = imutils.resize(image, width=1000)

        # otherwise, check to see if we should resize along the
        # height
        elif H > W and H > 1000:
            image = imutils.resize(image, height=1000)

        # prepare the image for detection
        (H, W) = image.shape[:2]
        output = image.copy()
        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        image = np.expand_dims(image, axis=0)

        # perform inference and compute the bounding boxes,
        # probabilities, and class labels
        (boxes, scores, labels, N) = sess.run(
            [boxesTensor, scoresTensor, classesTensor, numDetections],
            feed_dict={imageTensor: image})

        # squeeze the lists into a single dimension
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        labels = np.squeeze(labels)

        # max value of the label
        max_score_id = np.argmax(scores)
        best_label = labels[max_score_id]

        # rospy.logwarn("best_label: {0}".format(best_label))
        # rospy.logwarn("max(scores): {0}".format(max(scores)))

        if (max(scores) < 0.7):
            # rospy.logwarn("best_label: UNKNOWN")
            return TrafficLight.UNKNOWN
        elif (best_label == 1):
            # rospy.logwarn("best_label: GREEN")
            return TrafficLight.GREEN
        elif (best_label == 2):
            # rospy.logwarn("best_label: YELLOW")
            return TrafficLight.YELLOW
        elif (best_label == 3):
            # rospy.logwarn("best_label: RED")
            return TrafficLight.RED
            
        else:
            rospy.logwarn("best_label: UNKNOWN")
            return TrafficLight.UNKNOWN
