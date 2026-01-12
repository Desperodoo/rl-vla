import threading
import svar
import cv2
import time
import argparse
import h5py
import os
import numpy as np
import imageio
from carm import carm_api_py
import sys

messenger = svar.load('svar_messenger').messenger


class HiddenPrints:
    def __enter__(self):
        # 1. 刷新缓冲区，防止之前的输出被吞掉
        sys.stdout.flush()
        sys.stderr.flush()

        # 2. 备份原始的文件描述符
        self._original_stdout_fd = os.dup(sys.stdout.fileno())
        self._original_stderr_fd = os.dup(sys.stderr.fileno())

        # 3. 打开空设备 (Linux/Mac 是 /dev/null, Windows 是 NUL)
        self._devnull = os.open(os.devnull, os.O_WRONLY)

        # 4. 将标准输出和错误输出重定向到空设备
        os.dup2(self._devnull, sys.stdout.fileno())
        os.dup2(self._devnull, sys.stderr.fileno())

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 5. 恢复原始的文件描述符
        os.dup2(self._original_stdout_fd, sys.stdout.fileno())
        os.dup2(self._original_stderr_fd, sys.stderr.fileno())

        # 6. 关闭资源
        os.close(self._original_stdout_fd)
        os.close(self._original_stderr_fd)
        os.close(self._devnull)

def font_color(text, color="blue"):
    colors = {
        "blue": "\033[94m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "end": "\033[0m",
    }
    return f"{colors.get(color, colors['end'])}{text}{colors['end']}"


class RealEnviroment:
    def __init__(self, args):
        self.args        = args
        self.latest_imgs = None
        self.end_pose = None
        self.end_cmd = None
        self.joints = None
        self.joints_cmd = None
        self.joint_vel = None
        self.gripper_vel = None

        self.arm = carm_api_py.CArmApiWrapper(self.args.robot_ip)
        # self.arm.set_ready()
        # self.arm.set_control_mode(1)
        self.tau = 30

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

        self.freq = 200
        self.status_thread = threading.Thread(target=self.arm_status_thread)
        self.status_thread.start()
        self.plan_thread = threading.Thread(target=self.arm_plan)
        self.plan_thread.start()

        # subscribe sync images
        self.obs_lock = threading.Lock()
        self.sub_images  = messenger.subscribe("/cameras_sync_compressed",0,self.cbk_images_sync_compressed_func())

    def arm_status_thread(self):
        while True:
            self.publish_arm_state()
            time.sleep(1.0 / self.freq)
    
    def publish_arm_state(self):
        gripper = self.arm.get_gripper_pos()
        pose = self.arm.get_cart_pose()
        joint = self.arm.get_joint_pos()

        self.joint_vel = self.arm.get_joint_vel()
        self.gripper_vel = self.arm.get_gripper_vel()

        tau = self.arm.get_joint_tau()
        gripper_tau = self.arm.get_gripper_tau()
        
        self.end_pose = pose + [gripper]
        self.joints = joint+ [gripper]

    def arm_plan(self):
        while True:
            with HiddenPrints():
                joints_cmd = self.arm.get_plan_joint_pos().copy()
                self.end_cmd = self.arm.forward_kine(0, joints_cmd)[1]
                gripper = self.arm.get_gripper_pos()          # TODO: hack
                self.joints_cmd = joints_cmd+ [gripper] # TODO: hack
                self.end_cmd = self.end_cmd + [gripper]       # TODO: hack
            time.sleep(1.0 / self.freq)
    
    def decode_images(self, image_msgs, reverse_img=False):
        images = []
        stamp  = image_msgs[0]["header"]["stamp"]
        stamp  = stamp["sec"] + stamp["nanosec"] * 1e-9
        for idx, msg in enumerate(image_msgs):
            try:
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
            except:
                print("image snyc fail...........")
                return None
            
            if reverse_img:
                image = image[::-1]
            
            images.append(image)

            if self.args.vis:
                cv2.imshow(f"image_{idx}", image)
                cv2.waitKey(1)

        return stamp, images

    def get_observation(self):
        with self.obs_lock:
            if self.end_pose is None or self.joints is None:
                print("pose status is None")
                return None

            if self.latest_imgs is None:
                print("latest_imgs is None")
                return None
            
            # get qpos
            qpos = np.concatenate([self.joints, self.end_pose], axis=0)
            
            # get images
            if self.decode_images(self.latest_imgs) is None:
                return None
            else:
                stamp, images = self.decode_images(self.latest_imgs)

            return {"stamp" :   stamp, 
                    "images":   images,
                    "qpos":     qpos,
                    }

    def get_last_action(self):
        if self.joints_cmd is None or self.end_cmd is None:
            return None
        action = np.concatenate([self.joints_cmd, self.end_cmd], axis=0)
        return action

    def cbk_images_sync_compressed(self,msg):
        self.latest_imgs = msg

    def cbk_images_sync_compressed_func(self):
        def cbk(msg):
            try:
                return self.cbk_images_sync_compressed(msg)
            except Exception as e:  
                print("except:",e)
        return cbk 

def record_video(env, path, args):
    positions = []
    actions   = []
    fps = args.frame_rate
    step_time = 1.0/args.frame_rate

    # determine image resolution
    print("waiting observation stream.")
    while True:
        obs    = env.get_observation()
        if not obs is None:
            image  = np.concatenate(obs["images"], axis=0)
            print("camera numbers:", len(obs["images"]))
            print("camera size:", image.shape)
            break
        time.sleep(step_time)

    print("waiting action stream.")
    while True:
        time.sleep(step_time)
        action = env.get_last_action()
        if not action is None:
            break
        print("no action")

    print("capture of", path, "started")
    start_time = time.time()
    image_list = []
    for i in range(args.max_timesteps):
        video_path = f'{path}_video.mp4'
        delay = start_time + i * step_time - time.time()
        if delay > 0:   #30fps
            time.sleep(delay)
        obs    = env.get_observation()
        action = env.get_last_action()

        if obs is None or action is None:
            print("no observation or action")
            continue

        image = np.concatenate(obs["images"], axis=0)
        image_list.append(image)
        positions.append(obs["qpos"])
        actions.append(action)
        print(font_color(time.time(), color="blue"), 
              font_color(obs["stamp"], color="blue"), 
              font_color("recording:", color="blue"),
              font_color(i, color="blue")
            )
    
    image_np = np.array(image_list, dtype=np.uint8)
    writer = imageio.get_writer(video_path, fps=fps)
    for frame in image_np:
        writer.append_data(frame)
    writer.close()

    max_timesteps = len(actions)
    positions = np.array(positions)
    actions = np.array(actions)
    with h5py.File(path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = False
        obs_group  = root.create_group('observations')
        _ = obs_group.create_dataset('qpos', (max_timesteps, len(positions[0])))
        _ = root.create_dataset('action', (max_timesteps, len(actions[0])))
        print("qpos shape:", len(positions),len(positions[0]) )
        print("action shape:", len(actions),len(actions[0]) )
        root['/observations/qpos'][...] = positions
        root["/action"][...] = actions

    print(path, "saved.")
    exit()



def get_arguments():
    parser = argparse.ArgumentParser()
    # topic name of color image
    parser.add_argument('--camera_names', default='/camera_top/color/image_raw,/camera_arm/color/image_raw', 
                    type=str,  help='the camera joint topic list')
    parser.add_argument('--robot_ip', type=str, default='192.168.31.190')
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset_dir.', default="./data", required=True)
    parser.add_argument('--task_name', action='store', type=str, help='Task name.', default="aloha_mobile_dummy", required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', default=0, required=True)
    parser.add_argument('--max_timesteps', action='store', type=int, help='Max_timesteps.', default=500, required=False)
    parser.add_argument('--frame_rate', action='store', type=int, default=30)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument("--dds", type=str, default="svar_messenger_ros2", help="the dds plugin, default is svar_messenger_ros2, options: svar_zbus, svar_lcm")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()
    ros2 = svar.load(args.dds)
    subscriptions = []
    
    args.camera_names = args.camera_names.split(",")
    
    if len(args.camera_names) > 0:
        for camera in args.camera_names:
            subscriptions.append([camera, "sensor_msgs/msg/Image", 10])
        topic_sync = ros2.TopicSync({"sync_precision":0.05, "topics": args.camera_names,
                                     "topic_out":"/cameras_sync_compressed"})
    else:
        subscriptions.append(["/cameras_sync_compressed", "sensor_msgs/msg/CompressedImage", 10])
    print(subscriptions)
    transfer = ros2.Transfer({"node":"vr_data_recording",
                              "subscriptions":subscriptions})
    
    
    dataset_dir = os.path.join(args.dataset_dir, args.task_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        print(f"目录 '{dataset_dir}' 创建成功。")
    else:
        print(f"目录 '{dataset_dir}' 已存在。")

    env = RealEnviroment(args)
    path = os.path.join(dataset_dir, f"episode_{args.episode_idx}")
    record_video(env,path,args)
