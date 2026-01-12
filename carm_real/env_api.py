import threading
import svar
import cv2
import numpy as np
import time
import argparse
from scipy.spatial.transform import Rotation as R
import random
from carm import carm_api_py

tf = svar.load('svar_vectf')
vectf = svar.load('svar_quat_vectf')
messenger = svar.load('svar_messenger').messenger
ros2 = svar.load('svar_messenger_ros2')
cbor = svar.load('svar_cbor')

class RealEnviroment:
    def __init__(self, args):
        self.args = args

        self.arm = carm_api_py.CArmApiWrapper(self.args.robot_ip)
        self.arm.set_ready()
        self.arm.set_control_mode(self.args.robot_mode)
        self.tau = self.args.robot_tau

        arm_status = self.arm.get_status()
        print('----------------------------')
        print("arm_index: ", arm_status.arm_index)
        print("arm_name: ", arm_status.arm_name)
        print("arm_is_connected: ", arm_status.arm_is_connected)
        print("arm_dof: ", arm_status.arm_dof)
        print("servo_status: ", arm_status.servo_status)
        print("state: ", arm_status.state)
        print("speed_percentage: ", arm_status.speed_percentage)
        print("on_debug_mode: ", arm_status.on_debug_mode)
        print('arm_version: ', self.arm.get_version())
        print("gripper_tau: ", self.tau)
        print('----------------------------')

        not_origin = args.__dict__.get("not_origin",False)
        if not not_origin:
            self.init_status()
        
        self.end_state = None
        self.joint_state = None
        self.joint_vel = None
        self.gripper_vel = None
        self.gripper_tau  = None
        
        self.joint_cmd = None

        self.freq = 200
        self.status_thread = threading.Thread(target=self.arm_status_thread)
        self.status_thread.start()
        self.plan_thread = threading.Thread(target=self.arm_plan)
        self.plan_thread.start()

        # subscribe sync images
        self.obs_lock = threading.Lock()
        self.latest_imgs = None
        self.sub_images  = messenger.subscribe("/sync_cameras",0,self.cbk_images_sync_compressed_func())
    
    def arm_status_thread(self):
        while True:
            self.publish_arm_state()
            time.sleep(1.0 / self.freq)

    def arm_plan(self):
        while True:
            self.joint_cmd = self.arm.get_plan_joint_pos()
            time.sleep(1.0 / self.freq)
    
    def publish_arm_state(self):
        gripper = self.arm.get_gripper_pos()
        pose = self.arm.get_cart_pose()
        joint = self.arm.get_joint_pos()

        self.joint_vel = self.arm.get_joint_vel()
        self.gripper_vel = self.arm.get_gripper_vel()

        tau = self.arm.get_joint_tau()
        gripper_tau = self.arm.get_gripper_tau()
        
        self.end_state = pose + [gripper]
        self.joint_state = joint+ [gripper]

    def init_status(self):
        self.arm.set_gripper(self.args.arm_init_gripper, self.tau)
        # current_pose = self.arm.get_cart_pose()
        # current_pose[2] = current_pose[2]+0.02
        # self.arm.move_pose(current_pose)
        self.arm.move_pose(self.args.arm_init_pose)
        time.sleep(0.5)
    
    def end_control_nostep(self, action):
        self.arm.track_pose(action[:7])
        self.arm.set_gripper(action[-1],self.tau)
    
    def joint_control_nostep(self, action):
        self.arm.track_joint(action[:6])
        self.arm.set_gripper(action[-1],self.tau)
    
    def decode_images(self, image_msgs):
        images = []
        stamp  = image_msgs[0]["header"]["stamp"]
        stamp  = stamp["sec"] + stamp["nanosec"] * 1e-9
        for idx, msg in enumerate(image_msgs):
            if msg["encoding"] == "jpeg" or msg["encoding"] == "MJPG" or msg["encoding"] == "jpg" or msg["encoding"] == "png":
                image = cv2.imdecode(np.array(msg["data"]),cv2.IMREAD_COLOR)
            elif msg["encoding"] == "YUYV":
                image = np.frombuffer(msg["data"], dtype=np.uint8).reshape((msg["height"], msg["width"], 2))
                image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
            elif np.frombuffer(msg["data"], dtype=np.uint8).shape[0] ==307200:
                image = np.frombuffer(msg["data"], dtype=np.uint8).reshape((msg["height"], msg["width"], 1))
                image = np.repeat(image, repeats=3, axis=-1)
            else:
                image = np.frombuffer(msg["data"], dtype=np.uint8).reshape((msg["height"], msg["width"], 3))
            images.append(image)

            if self.args.vis:
                cv2.imshow(f"image_{idx}", image)
                cv2.waitKey(1)

        return stamp, images

    def get_observation(self):
        
        with self.obs_lock:
            if self.end_state is None :
                print("end_state is None")
                return None
            
            if self.joint_state is None :
                print("joint_state is None")
                return None
            
            if self.latest_imgs is None:
                print("latest_imgs is None")
                return None
            
            # get qpos
            qpos_joint = self.joint_state
            qpos_end = self.end_state
            
            # get images
            stamp, images = self.decode_images(self.latest_imgs)
            
            return {"stamp" :     stamp, 
                    "images":     images,
                    "qpos_joint": qpos_joint,
                    "qpos_end":   qpos_end
                    }
    
    def cbk_images_sync_compressed(self,msg):
        self.latest_imgs = msg
 
    def cbk_images_sync_compressed_func(self):
        def cbk(msg):
            try:
                return self.cbk_images_sync_compressed(msg)
            except Exception as e:  
                print("except:",e)
        return cbk 