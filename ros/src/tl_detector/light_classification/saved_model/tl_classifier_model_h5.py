#!/usr/bin/env python
import rospy
import cv2
import h5py
import tensorflow as tf
from keras.models import load_model
from styx_msgs.msg import TrafficLight

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        # pass		
	self.model = load_model('/home/student/Desktop/CarND-Capstone/ros/src/tl_detector/light_classification/saved_model/model.h5')
	self.graph = tf.get_default_graph()
		
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
	model = self.model
	graph = self.graph

	with graph.as_default():
	    if image[0][0][0]:
		img = image
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = img[60:135,:,:]
		img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
		img = cv2.GaussianBlur(img,  (3, 3), 0)
		img = cv2.resize(img, (200, 66))
		img = img/255

		traffic_light_color = float(model.predict(img[None, :, :, :], batch_size=1))			
		best_label = int(round(traffic_light_color))

		rospy.logwarn("best_label: {0}".format(best_label))
		rospy.logwarn("score     : {0}".format(traffic_light_color))

		if (best_label == 0):
		    return TrafficLight.RED
		elif (best_label == 1):
		    return TrafficLight.YELLOW
		elif (best_label == 2):
		    return TrafficLight.GREEN
		elif (best_label == 4):
		    return TrafficLight.UNKNOWN
		else:
		    return TrafficLight.UNKNOWN

	    else:
		return TrafficLight.UNKNOWN
