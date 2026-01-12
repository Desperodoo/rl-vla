#!/usr/bin/env python3
"""
ROS 多相机图像同步器
使用 message_filters.ApproximateTimeSynchronizer 实现多相机时间同步
替代 svar 的 ros2.TopicSync 功能
"""

import rospy
import threading
import numpy as np
import cv2
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import message_filters


class ImageSynchronizer:
    """
    多相机图像同步器
    支持多个相机话题的时间同步订阅
    """
    
    def __init__(self, camera_topics, sync_slop=0.1, queue_size=10, use_compressed=False):
        """
        初始化图像同步器
        
        Args:
            camera_topics: 相机话题列表，如 ["/camera/color/image_raw"]
            sync_slop: 时间同步容差（秒）
            queue_size: 消息队列大小
            use_compressed: 是否使用压缩图像话题
        """
        self.camera_topics = camera_topics
        self.sync_slop = sync_slop
        self.queue_size = queue_size
        self.use_compressed = use_compressed
        
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_images = None
        self.latest_stamp = None
        
        self._setup_subscribers()
    
    def _setup_subscribers(self):
        """设置同步订阅器"""
        if len(self.camera_topics) == 0:
            rospy.logwarn("No camera topics provided")
            return
        
        # 确定消息类型
        msg_type = CompressedImage if self.use_compressed else Image
        
        # 创建订阅器列表
        self.subscribers = []
        for topic in self.camera_topics:
            sub = message_filters.Subscriber(topic, msg_type)
            self.subscribers.append(sub)
            rospy.loginfo(f"Subscribing to: {topic}")
        
        # 创建时间同步器
        if len(self.subscribers) == 1:
            # 单相机直接订阅
            self.subscribers[0].registerCallback(self._single_callback)
        else:
            # 多相机使用 ApproximateTimeSynchronizer
            self.sync = message_filters.ApproximateTimeSynchronizer(
                self.subscribers,
                queue_size=self.queue_size,
                slop=self.sync_slop
            )
            self.sync.registerCallback(self._sync_callback)
    
    def _single_callback(self, msg):
        """单相机回调"""
        try:
            stamp = msg.header.stamp.to_sec()
            image = self._decode_image(msg)
            
            with self.lock:
                self.latest_images = [image]
                self.latest_stamp = stamp
        except Exception as e:
            rospy.logerr(f"Error in single callback: {e}")
    
    def _sync_callback(self, *msgs):
        """多相机同步回调"""
        try:
            # 使用第一个消息的时间戳
            stamp = msgs[0].header.stamp.to_sec()
            
            images = []
            for msg in msgs:
                image = self._decode_image(msg)
                images.append(image)
            
            with self.lock:
                self.latest_images = images
                self.latest_stamp = stamp
                
        except Exception as e:
            rospy.logerr(f"Error in sync callback: {e}")
    
    def _decode_image(self, msg):
        """解码图像消息"""
        if isinstance(msg, CompressedImage):
            # 压缩图像
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            # 原始图像
            if msg.encoding == "rgb8":
                image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            elif msg.encoding == "bgr8":
                image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            elif msg.encoding == "mono8":
                image = self.bridge.imgmsg_to_cv2(msg, "mono8")
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif msg.encoding == "16UC1":
                # 深度图像
                image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            else:
                image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        return image
    
    def get_images(self):
        """
        获取最新的同步图像
        
        Returns:
            tuple: (timestamp, images_list) 或 (None, None) 如果没有图像
        """
        with self.lock:
            if self.latest_images is None:
                return None, None
            return self.latest_stamp, self.latest_images.copy()
    
    def wait_for_images(self, timeout=5.0):
        """
        等待图像到达
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            bool: 是否成功获取图像
        """
        start_time = rospy.Time.now()
        rate = rospy.Rate(10)
        
        while not rospy.is_shutdown():
            stamp, images = self.get_images()
            if images is not None:
                return True
            
            if (rospy.Time.now() - start_time).to_sec() > timeout:
                rospy.logwarn("Timeout waiting for images")
                return False
            
            rate.sleep()
        
        return False


class SingleImageSubscriber:
    """
    单相机订阅器（简化版）
    用于不需要多相机同步的场景
    """
    
    def __init__(self, topic, use_compressed=False):
        """
        初始化单相机订阅器
        
        Args:
            topic: 相机话题
            use_compressed: 是否使用压缩图像
        """
        self.topic = topic
        self.use_compressed = use_compressed
        
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_image = None
        self.latest_stamp = None
        
        msg_type = CompressedImage if use_compressed else Image
        self.subscriber = rospy.Subscriber(topic, msg_type, self._callback, queue_size=1)
        rospy.loginfo(f"Subscribing to: {topic}")
    
    def _callback(self, msg):
        """图像回调"""
        try:
            stamp = msg.header.stamp.to_sec()
            
            if isinstance(msg, CompressedImage):
                np_arr = np.frombuffer(msg.data, np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                if msg.encoding == "rgb8":
                    image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                else:
                    image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            with self.lock:
                self.latest_image = image
                self.latest_stamp = stamp
                
        except Exception as e:
            rospy.logerr(f"Error in image callback: {e}")
    
    def get_image(self):
        """获取最新图像"""
        with self.lock:
            if self.latest_image is None:
                return None, None
            return self.latest_stamp, self.latest_image.copy()
