#!/usr/bin/env python3
import rospy
import cv2
import pyrealsense2 as rs
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

class CameraNode:
    def __init__(self):
        rospy.init_node('multi_cam_node')
        self.bridge = CvBridge()

        # Configure first RealSense camera (using serial number)
        self.pipeline_1 = rs.pipeline()
        config_1 = rs.config()
        config_1.enable_device('338622073582')  # First camera serial number
        config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline_1.start(config_1)

        # Configure second RealSense camera (using serial number)
        self.pipeline_2 = rs.pipeline()
        config_2 = rs.config()
        config_2.enable_device('148522073685')  # Second camera serial number
        config_2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline_2.start(config_2)

        # Publish image topics
        self.pub_1 = rospy.Publisher('/cam_1', Image, queue_size=30)
        self.pub_2 = rospy.Publisher('/cam_2', Image, queue_size=30)

        # Timer to publish images at 10Hz frequency
        rospy.Timer(rospy.Duration(0.1), self.publish_images)  # 10Hz adjustable

    def publish_images(self, event):
        # Get image data from first camera
        frames_1 = self.pipeline_1.wait_for_frames()  # Get frames
        color_frame_1 = frames_1.get_color_frame()  # Get color frame
        if not color_frame_1:
            rospy.logwarn("Failed to capture image from camera 1.")
        else:
            frame_1 = np.asanyarray(color_frame_1.get_data())  # Convert to NumPy array
            ros_img_1 = self.bridge.cv2_to_imgmsg(frame_1, encoding='bgr8')
            self.pub_1.publish(ros_img_1)
            rospy.loginfo("Published image from camera 1")
            
            # Visualize the image from camera 1
            # self.visualize_image(frame_1, "Camera 1")

        # Get image data from second camera
        frames_2 = self.pipeline_2.wait_for_frames()  # Get frames
        color_frame_2 = frames_2.get_color_frame()  # Get color frame
        if not color_frame_2:
            rospy.logwarn("Failed to capture image from camera 2.")
        else:
            frame_2 = np.asanyarray(color_frame_2.get_data())  # Convert to NumPy array
            ros_img_2 = self.bridge.cv2_to_imgmsg(frame_2, encoding='bgr8')
            self.pub_2.publish(ros_img_2)
            rospy.loginfo("Published image from camera 2")
            
            # Visualize the image from camera 2
            # self.visualize_image(frame_2, "Camera 2")

    def visualize_image(self, image, window_name):
        # Display the image using OpenCV
        cv2.imshow(window_name, image)
        cv2.waitKey(1)  # Wait for a key press to update the window

    def __del__(self):
        # Stop streams
        self.pipeline_1.stop()
        self.pipeline_2.stop()
        rospy.loginfo("Cameras released.")
        cv2.destroyAllWindows()  # Close all OpenCV windows when done

if __name__ == '__main__':
    try:
        CameraNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
