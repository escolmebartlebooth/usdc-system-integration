from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import rospy
import datetime

CONFIDENCE_CUTOFF = 0.5

class TLClassifier(object):
    def __init__(self, model=None):
        #TODO load classifier
        self.GRAPH_FILE = model
        
        # load the graph one time
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.GRAPH_FILE, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        
            # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')

            # The classification of the object (integer id).
            self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                        
        self.sess = tf.Session(graph=self.detection_graph)
        
        
    def filter_boxes(self, min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)

        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        with self.detection_graph.as_default():
            image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
            start = datetime.datetime.now()
            (boxes, scores, classes) = self.sess.run([self.detection_boxes, self.detection_scores,                                                                                      self.detection_classes],
                                                     feed_dict={self.image_tensor: image_np})
            end = datetime.datetime.now()
            # rospy.logwarn("detection time: {0}".format((end-start).total_seconds()))
            # Remove unnecessary dimensions
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)

            # Filter boxes with a confidence score less than `confidence_cutoff`
            boxes, scores, classes = self.filter_boxes(CONFIDENCE_CUTOFF, boxes, scores, classes)
            if len(classes) > 0:
                # rospy.logwarn("matches {0} and class: {1}".format(len(classes),                                                                                                                       classes[0]))
                if classes[0] == 1: 
                    return TrafficLight.GREEN
                elif classes[0] == 2: 
                    return TrafficLight.RED
                elif classes[0] == 3: 
                    return TrafficLight.YELLOW
                else:
                    return TrafficLight.UNKNOWN 
            else:
                return TrafficLight.UNKNOWN
