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
    
    def __init__(self, camera_topics, sync_slop=0.02, queue_size=10, use_compressed=False,
                 target_width=None, target_height=None):
        """
        初始化图像同步器
        
        Args:
            camera_topics: 相机话题列表，如 ["/camera/color/image_raw"]
            sync_slop: 时间同步容差（秒）
            queue_size: 消息队列大小
            use_compressed: 是否使用压缩图像话题
            target_width: 目标图像宽度（None 表示不缩放）
            target_height: 目标图像高度（None 表示不缩放）
        """
        self.camera_topics = camera_topics
        self.sync_slop = sync_slop
        self.queue_size = queue_size
        self.use_compressed = use_compressed
        self.target_width = target_width
        self.target_height = target_height
        
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_images = None
        self.latest_stamp = None
        
        if target_width and target_height:
            rospy.loginfo(f"Image resize enabled: {target_width}x{target_height}")
        
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
        """
        解码图像消息 - 使用纯 numpy 避免 cv_bridge 的 libffi 冲突
        
        【重要】统一返回 RGB 格式：
        - RealSense ROS 节点发布 RGB8 格式
        - 本函数始终返回 RGB 格式（与大多数深度学习框架兼容）
        - 使用 OpenCV 显示时需要手动转换为 BGR
        """
        if isinstance(msg, CompressedImage):
            # 压缩图像 - cv2.imdecode 返回 BGR，需要转换为 RGB
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        else:
            # 原始图像 - 使用纯 numpy 解码，避免 cv_bridge
            # 这样可以避免 conda libffi 与系统库冲突
            height = msg.height
            width = msg.width
            
            if msg.encoding == "rgb8":
                # RealSense 发布的是 RGB8，保持原样
                image = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width, 3)
                image = image.copy()  # 确保可写
            elif msg.encoding == "bgr8":
                # BGR 转换为 RGB
                image = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width, 3)
                image = image[:, :, ::-1].copy()  # BGR -> RGB
            elif msg.encoding == "mono8":
                # 灰度图转换为 RGB（3 通道）
                image = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif msg.encoding == "16UC1":
                # 深度图像保持原样（单通道）
                image = np.frombuffer(msg.data, dtype=np.uint16).reshape(height, width)
            elif msg.encoding == "rgba8":
                image = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width, 4)
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif msg.encoding == "bgra8":
                image = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width, 4)
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                # 默认尝试作为 RGB
                try:
                    image = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width, 3)
                    image = image.copy()
                    rospy.logwarn_once(f"Unknown encoding: {msg.encoding}, treating as RGB")
                except ValueError:
                    rospy.logwarn(f"Unknown encoding: {msg.encoding}, falling back to cv_bridge")
                    image = self.bridge.imgmsg_to_cv2(msg, "rgb8")  # 请求 RGB 格式
        
        return image
    
    def _resize_image(self, image):
        """缩放图像到目标尺寸"""
        if self.target_width is None or self.target_height is None:
            return image
        
        h, w = image.shape[:2]
        if w == self.target_width and h == self.target_height:
            return image
        
        return cv2.resize(image, (self.target_width, self.target_height), 
                         interpolation=cv2.INTER_AREA)
    
    def get_images(self):
        """
        获取最新的同步图像
        
        Returns:
            tuple: (timestamp, images_list) 或 (None, None) 如果没有图像
        """
        with self.lock:
            if self.latest_images is None:
                return None, None
            # 返回缩放后的图像
            resized_images = [self._resize_image(img) for img in self.latest_images]
            return self.latest_stamp, resized_images
    
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
