import modules
import argparse
import torch
import numpy as np
import time
from einops import rearrange
import svar
import threading
import real.carm_real.env_api
import cv2
import json
from scipy.spatial.transform import Rotation as R

messenger = svar.load('svar_messenger').messenger
vectf = svar.load('svar_vectf')

def pose_to_transform_matrix(position, quaternion):
      """
      将位姿 (xyz + 四元数) 转换为 4x4 变换矩阵
      :param position: 平移 [x, y, z]
      :param quaternion: 四元数 [qx, qy, qz, qw]
      :return: 4x4 变换矩阵
      """
      # 创建旋转矩阵
      rotation = R.from_quat(quaternion).as_matrix()
      # 创建 4x4 变换矩阵
      transform = np.eye(4)
      transform[:3, :3] = rotation
      transform[:3, 3] = position
      return transform

def compute_relative_pose(pose_absolute, pose_init, gripper):
    """
    计算相对位姿
    参数：
        pose_absolute: 绝对位姿矩阵，形状为(N, 7),其中N为帧数
        (x,y,z,w,x,y,z)表示平移和旋转
    返回：
        pose_relative: 相对位姿矩阵形状与pose_absolute相同
    """
   
    strat2current = pose_to_transform_matrix(pose_absolute[:3], pose_absolute[3:])
    start = pose_to_transform_matrix(pose_init[:3], pose_init[3:])
       
     
    current2global = start @ strat2current
        
    cur_position = current2global[:3, 3]
    cur_euler =R.from_matrix(current2global[:3, :3]).as_quat()
    
    pose_relative = cur_position.tolist() + cur_euler.tolist() + [gripper]
    # 返回计算得到的相对位姿矩阵
    
    return pose_relative


class InferenceReal:
    def __init__(self, args):
        self.temporal_factor_k = args.__dict__.get("temporal_factor_k", 0.01)

        self.env        = real.carm_real.env_api.RealEnviroment(args)
        self.latest_obs = None
        self.args       = args
        self.action_tfs = []
        self.lock_tfs   = threading.Lock()
        self.policy     = modules.make_policy(args)
    
        self.thread_inference = threading.Thread(target=self.inference_thread)
        self.thread_inference.start()

    def normalize_images(self, obs):
        curr_images = []
        for index, image in enumerate(obs["images"]):
            curr_image = rearrange(image, 'h w c -> c h w')
            curr_images.append(curr_image)
        #     curr_images.append(image) 
        # curr_image = rearrange(np.concatenate(curr_images, axis=0), 'h w c -> c h w')[None,:,:,:] 
        curr_image = np.stack(curr_images, axis=0)        
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
        curr_image = curr_image[:,0].unsqueeze(1) # 1, 1, h, w
        return curr_image

        
    def inference_thread(self):
        print("inference thread started.")
        desire_inference_freq = self.args.__dict__.get("desire_inference_freq", 20)
        desire_period = 1.0 / desire_inference_freq
        pos_lookahead_step = self.args.__dict__.get("pos_lookahead_step", 1)
        pos_lookahead_duration = self.args.__dict__.get("pos_lookahead_duration", 0.015)
        joint_cmd_mode = self.args.__dict__.get("joint_cmd_mode", False)

        pos_lookahead_step_start_idx = 0
        with torch.inference_mode():
            while 1:
                self.latest_obs = self.env.get_observation()
                if self.latest_obs is None:
                    time.sleep(0.5)
                    print("wait observation")
                    continue
                
                last_start = time.time()
                print("g3 start")
            
                # qpos
                qpos_joint = np.array(self.latest_obs['qpos_joint'])
                qpos_end = np.array(self.latest_obs['qpos_end']).tolist()
                qpos = torch.from_numpy(qpos_joint).float().cuda().unsqueeze(0)
                
                # images
                curr_image = self.normalize_images(self.latest_obs)

                # inference
                ret = self.policy({"qpos":qpos, "image":curr_image})
                all_actions = ret["a_hat"].squeeze(0).cpu().numpy()
                
                if not joint_cmd_mode:
                    all_endactions = []
                    for i in range(all_actions.shape[0]):
                        end_action = all_actions[i][7:]
                        grip = all_actions[i][6]
                        action = compute_relative_pose(end_action[:7], qpos_end[:7], grip)
                        all_endactions.append(action)
                    all_actions = np.array(all_endactions)

                stamp = self.latest_obs["stamp"]
                tf = vectf.VecTF({})
                
                pos_lookahead_step_start_idx += 1
                for i in range(0, len(all_actions)):
                    if pos_lookahead_step == 1:
                        tf.append(stamp + i * desire_period, all_actions[i].tolist())
                    else:
                        if pos_lookahead_step_start_idx % pos_lookahead_step == 0:
                            tf.append(stamp + i * desire_period, all_actions[i].tolist())
                        else:
                            tf.append(stamp + i * pos_lookahead_duration, all_actions[i].tolist())

                with self.lock_tfs:
                    self.action_tfs.append(tf)
                     
                print("inference time:", time.time() - last_start)

                wait_tm = desire_period - (time.time() - last_start)
                if wait_tm > 0:
                    time.sleep(wait_tm)

    def control_loop(self):

        joint_cmd_mode = self.args.__dict__.get("joint_cmd_mode", False)

        while 1:
            # interpolate actions
            action_candidates = []
            tm = time.time()
            valid_offset = 0

            with self.lock_tfs: # obtain candidates and update tfs
                for index,tf in enumerate(self.action_tfs):
                    action_candidate = tf.get_once(tm)
                    if action_candidate is None:
                        valid_offset = index # throwout
                        continue

                    action_candidates.append(action_candidate)
                self.action_tfs = self.action_tfs[valid_offset:]

            if len(action_candidates) < 1:
                time.sleep(0.02)
                continue
            
            all_actions = np.array(action_candidates) 
            exp_weights = np.exp(-self.temporal_factor_k * np.arange(len(action_candidates)-1, -1, -1))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = exp_weights[:, np.newaxis] # expend dim
            action = (all_actions * exp_weights).sum(axis=0, keepdims=True)

            if joint_cmd_mode:
                print("joint control")
                self.env.joint_control_nostep(action[0])
            else:
                print("endpose control")
                self.env.end_control_nostep(action[0])

            time.sleep(0.005)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    
    parser.add_argument('--robot_ip', type=str, default='10.42.0.101')
    parser.add_argument('--robot_mode', type=int, default=1)
    parser.add_argument('--robot_tau', type=float, default=10)
    parser.add_argument('--arm_init_pose', type=list, default=[0.26, -0.02, 0.22, 1,0, 0, 0])
    parser.add_argument('--arm_init_gripper', type=list, default=0.05)
    parser.add_argument('--pretrain', default='', type=str, help="The pretrained model")
    parser.add_argument('--desire_inference_freq', default=30, type=float, help='the desire inference frequency')
    parser.add_argument('--temporal_factor_k', default=0.05, type=float, help='the tepmoral factor')
    parser.add_argument('--vis', default=False, action="store_true",  help="Visualize or not, default vis")
    parser.add_argument('--camera_names', default='/camera_arm/color/image_raw', type=str,  help='visualize or not')
    parser.add_argument('--pos_lookahead_step', action='store', type=int, help='pos_lookahead_step',
                        default=1,required=False)
    parser.add_argument('--pos_lookahead_duration', action='store', type=float, help='pos_lookahead_step',
                        default=0.015,required=False)
    parser.add_argument("--not_origin", action="store_true", help="Enable griper enlage")
    parser.add_argument("--joint_cmd_mode", action="store_true", help="Enable griper enlage")
    parser.add_argument('--dds', default="svar_zbus", type=str, help='the dds')
    args = parser.parse_args()
    
    ros2 = svar.load(args.dds)

    subscriptions = []
    args.camera_names = args.camera_names.split(",")
    for camera in args.camera_names:
        subscriptions.append([camera, "sensor_msgs/msg/Image", 10])

    topic_sync = ros2.TopicSync({"sync_precision": 0.05, 
                                 "topics":         args.camera_names,
                                 "topic_out":      "/sync_cameras"})

    transfer = ros2.Transfer({"node":          "inference_real",
                              "subscriptions": subscriptions,
                            })

    inference = InferenceReal(args)
    inference.control_loop()