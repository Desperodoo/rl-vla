#!/usr/bin/env python3
"""
RealSense D405 ROS 话题订阅示例
用于在ROS中订阅相机图像并处理
"""

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np


class RealsenseSubscriber:
    """RealSense D405 ROS话题订阅器"""
    
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('realsense_subscriber', anonymous=True)
        
        # 初始化CV Bridge
        self.bridge = CvBridge()
        
        # 存储最新图像
        self.color_image = None
        self.depth_image = None
        self.color_info = None
        self.depth_info = None
        
        # 订阅话题
        self.color_sub = rospy.Subscriber(
            '/camera/color/image_raw',
            Image,
            self.color_callback
        )
        self.depth_sub = rospy.Subscriber(
            '/camera/depth/image_rect_raw',
            Image,
            self.depth_callback
        )
        self.color_info_sub = rospy.Subscriber(
            '/camera/color/camera_info',
            CameraInfo,
            self.color_info_callback
        )
        self.depth_info_sub = rospy.Subscriber(
            '/camera/depth/camera_info',
            CameraInfo,
            self.depth_info_callback
        )
        
        rospy.loginfo("RealSense D405 订阅器已启动")
        rospy.loginfo("订阅话题:")
        rospy.loginfo("  - /camera/color/image_raw")
        rospy.loginfo("  - /camera/depth/image_rect_raw")
        rospy.loginfo("  - /camera/color/camera_info")
        rospy.loginfo("  - /camera/depth/camera_info")
    
    def color_callback(self, msg):
        """彩色图像回调"""
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"彩色图像转换失败: {e}")
    
    def depth_callback(self, msg):
        """深度图像回调"""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        except Exception as e:
            rospy.logerr(f"深度图像转换失败: {e}")
    
    def color_info_callback(self, msg):
        """彩色相机内参回调"""
        self.color_info = msg
    
    def depth_info_callback(self, msg):
        """深度相机内参回调"""
        self.depth_info = msg
    
    def get_color_image(self):
        """获取最新彩色图像"""
        return self.color_image
    
    def get_depth_image(self):
        """获取最新深度图像"""
        return self.depth_image
    
    def get_depth_in_meters(self):
        """获取深度图 (单位: 米)"""
        if self.depth_image is not None:
            return self.depth_image.astype(np.float32) * 0.0001  # D405深度比例
        return None
    
    def get_intrinsics(self):
        """获取相机内参"""
        intrinsics = {}
        if self.color_info:
            intrinsics['color'] = {
                'fx': self.color_info.K[0],
                'fy': self.color_info.K[4],
                'cx': self.color_info.K[2],
                'cy': self.color_info.K[5],
                'width': self.color_info.width,
                'height': self.color_info.height
            }
        if self.depth_info:
            intrinsics['depth'] = {
                'fx': self.depth_info.K[0],
                'fy': self.depth_info.K[4],
                'cx': self.depth_info.K[2],
                'cy': self.depth_info.K[5],
                'width': self.depth_info.width,
                'height': self.depth_info.height
            }
        return intrinsics
    
    def visualize(self, show_depth=True, show_color=True):
        """可视化图像"""
        if show_color and self.color_image is not None:
            cv2.imshow('Color Image', self.color_image)
        
        if show_depth and self.depth_image is not None:
            # 深度图着色
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(self.depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
            cv2.imshow('Depth Image', depth_colormap)
        
        return cv2.waitKey(1)


def main():
    """主函数 - 演示ROS话题订阅"""
    subscriber = RealsenseSubscriber()
    
    rate = rospy.Rate(30)  # 30Hz
    
    rospy.loginfo("按 'q' 退出")
    
    while not rospy.is_shutdown():
        # 可视化
        key = subscriber.visualize()
        if key == ord('q'):
            break
        
        # 打印内参信息 (每5秒一次)
        if rospy.Time.now().secs % 5 == 0:
            intrinsics = subscriber.get_intrinsics()
            if intrinsics:
                rospy.loginfo_throttle(5, f"相机内参: {intrinsics}")
        
        rate.sleep()
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
