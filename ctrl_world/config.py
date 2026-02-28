import torch
import os
import json
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class wm_args:
    ########################### training args ##############################
    # model paths
    svd_model_path = "/cephfs/shared/llm/stable-video-diffusion-img2vid"
    clip_model_path = "/cephfs/shared/llm/clip-vit-base-patch32"
    ckpt_path = '/cephfs/cjyyj/code/video_evaluation/output2/exp33_210_s11/checkpoint-10000.pt'
    pi_ckpt = '/cephfs/shared/llm/openpi/openpi-assets-preview/checkpoints/pi05_droid'

    # dataset parameters
    # raw data
    dataset_root_path = "dataset_example"
    dataset_names = 'droid_subset'
    # meta info
    dataset_meta_info_path = 'dataset_meta_info' #'/cephfs/cjyyj/code/video_evaluation/exp_cfg'#'dataset_meta_info'
    dataset_cfgs = dataset_names
    prob=[1.0]
    annotation_name='annotation' #'annotation_all_skip1'
    num_workers=4
    down_sample=3 # downsample 15hz to 5hz
    skip_step = 1
    

    # logs parameters
    debug = False
    tag = 'doird_subset'
    output_dir = f"model_ckpt/{tag}"
    wandb_run_name = tag
    wandb_project_name = "droid_example"


    # training parameters
    learning_rate= 1e-5 # 5e-6
    gradient_accumulation_steps = 1
    mixed_precision = 'fp16'
    train_batch_size = 4
    shuffle = True
    num_train_epochs = 100
    max_train_steps = 500000
    checkpointing_steps = 20000
    validation_steps = 2500
    max_grad_norm = 1.0
    log_every_n_steps = 100  # VLAW MODIFICATION: 可配置日志记录频率
    # for val
    video_num= 10

    ############################ model args ##############################

    # model parameters
    motion_bucket_id = 127
    fps = 7
    guidance_scale = 2 #7.5 #7.5 #7.5 #3.0
    num_inference_steps = 50
    decode_chunk_size = 7
    width = 320
    height = 192
    # num history and num future predictions
    num_frames= 5
    num_history = 6
    action_dim = 7
    text_cond = True
    frame_level_cond = True
    his_cond_zero = False
    dtype = torch.bfloat16 # [torch.float32, torch.bfloat16] # during inference, we can use bfloat16 to accelerate the inference speed and save memory



    ########################### rollout args ############################
    # policy
    task_type: str = "pickplace" # choose from ['pickplace', 'towel_fold', 'wipe_table', 'tissue', 'close_laptop','tissue','drawer','stack']
    gripper_max_dict = {'replay':1.0, 'pickplace':0.75, 'towel_fold':0.95, 'wipe_table':0.95, 'tissue':0.97, 'close_laptop':0.95,'drawer':0.75,'stack':0.75,}
    ##############################################################################
    policy_type = 'pi05' # choose from ['pi05', 'pi0', 'pi0fast']
    action_adapter = 'models/action_adapter/model2_15_9.pth' # adapat action from joint vel to cartesian pose
    pred_step = 5 # predict 5 steps (1s) action each time
    policy_skip_step = 2 # horizon = (pred_step-1) * policy_skip_step
    interact_num = 12 # number of interactions (each interaction contains pred_step steps)

    # wm
    data_stat_path = 'dataset_meta_info/droid/stat.json'
    val_model_path = ckpt_path
    history_idx = [0,0,-12,-9,-6,-3]

    # save
    save_dir = 'synthetic_traj'

    # select different traj for different tasks
    def __post_init__(self):
        # Per-task gripper max
        self.gripper_max = self.gripper_max_dict.get(self.task_type, 0.75)
        # Default task_name
        self.task_name = f"Rollouts_interact_pi"
        if self.task_type == "replay":
            self.task_name = "Rollouts_replay"

        # Configure per-task eval sets
        if self.task_type == "replay":
            self.val_dataset_dir = "dataset_example/droid_subset"
            self.val_id = ["899", "18599","199",]
            self.start_idx = [8, 14, 8] * len(self.val_id)
            self.instruction = [""] * len(self.val_id)
            self.task_name = "Rollouts_replay"

        elif self.task_type == "keyboard":
            self.val_dataset_dir = "dataset_example/droid_subset"
            self.val_id = ["1799"]
            self.start_idx = [23] * len(self.val_id)
            self.instruction = [""] * len(self.val_id)
            self.task_name = "Rollouts_keyboard"

        # elif self.task_type == "keyboard2":
        #     self.val_dataset_dir = "/cephfs/shared/droid_hf/droid_svd_v2"
        #     self.val_id = ["1499"]*100
        #     self.start_idx = [8] * len(self.val_id) # 2599 8 #9499 10
        #     self.instruction = [""] * len(self.val_id)
        #     self.task_name = "Rollouts_keyboard_1499"
        #     self.ineraction_num = 7

        elif self.task_type == "pickplace":
            self.interact_num = 15
            self.val_dataset_dir = "dataset_example/droid_new_setup"
            self.val_id = ['0001','0002','0003']
            self.start_idx = [0] * len(self.val_id)
            self.instruction = [
                "pick up the green block and place in plate",
                "pick up the green block and place in plate",
                "pick up the blue block and place in plate",]

        elif self.task_type == "towel_fold":
            self.interact_num = 15
            self.val_dataset_dir = "dataset_example/droid_new_setup"
            self.val_id =['0004','0005']
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ["fold the towel"] * len(self.val_id)

        elif self.task_type == "wipe_table":
            self.val_dataset_dir = "dataset_example/droid_new_setup"
            self.val_id = ['0006','0007']
            self.start_idx = [0] * len(self.val_id)
            self.instruction = [
                "move the towel from left to right",
                "move the towel from left to right"
            ]

        elif self.task_type == "tissue":
            self.interact_num = 10
            self.val_dataset_dir = "dataset_example/droid_new_setup"
            self.val_id = ['0008','0009']
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ["pull one tissue out of the box"] * len(self.val_id)
            self.policy_skip_step = 3

        elif self.task_type == "close_laptop":
            self.val_dataset_dir = "dataset_example/droid_new_setup"
            self.val_id = ['0010','0011']
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ["close the laptop"] * len(self.val_id)
            self.policy_skip_step = 3

        elif self.task_type == "stack":
            self.val_dataset_dir = "dataset_example/droid_new_setup"
            self.val_id = ['0012','0013']
            self.start_idx = [5] * len(self.val_id)
            self.instruction = ["stack the blue block on the red block"] * len(self.val_id)
        
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")


# VLAW MODIFICATION: ManiSkill 适配配置
# 与原版 wm_args 的关键差异:
#   - 2 相机垂直拼接 (高度384), 非 3 相机 (高度576)
#   - latent: (T, 4, 48, 24)  vs DROID (T, 4, 72, 40)
#   - action: delta pose (增量), 非 DROID 绝对位姿
#   - action 归一化使用 ManiSkill 自己的统计量
@dataclass
class wm_args_maniskill(wm_args):
    """Ctrl-World 针对 ManiSkill 环境的适配配置."""

    # ---- 模型路径 (覆盖 wm_args 的 cephfs 路径) ----
    # VLAW MODIFICATION: 使用本地下载的预训练权重
    svd_model_path: str = "../checkpoints/vlaw/world_model/pretrained/stable-video-diffusion-img2vid"
    clip_model_path: str = "../checkpoints/vlaw/world_model/pretrained/clip-vit-base-patch32"
    ckpt_path: Optional[str] = "../checkpoints/vlaw/world_model/pretrained/Ctrl-World/checkpoint-10000.pt"

    # ---- 图像分辨率 ----
    # VLAW MODIFICATION: 2 相机 × 192px = 384px 高度 (vs DROID 3×192=576)
    width: int = 192       # 单相机宽度 (pixels)
    height: int = 384      # 双相机垂直拼接: 2 × 192

    # ---- 序列长度 ----
    num_frames: int = 5    # 预测帧数 (同 DROID)
    num_history: int = 4   # 历史帧数 (DROID=6, 降至4节省显存)

    # ---- 动作 ----
    # VLAW MODIFICATION: ManiSkill 使用 delta pose (增量), 7D
    action_dim: int = 7    # xyz_delta(3) + euler_delta(3) + gripper(1)

    # ---- 数据 ----
    # VLAW MODIFICATION: ManiSkill HDF5 数据路径
    dataset_root_path: str = "../data/vlaw"
    dataset_names: str = "demos"             # 与 data_collector 输出路径对应
    dataset_meta_info_path: str = "../data/vlaw/meta_info"
    annotation_name: str = "annotation"
    down_sample: int = 1   # ManiSkill 数据已经是目标频率, 无需额外下采样
    skip_step: int = 1

    # ---- 归一化统计量路径 ----
    # VLAW MODIFICATION: 使用 ManiSkill 自己的 stat.json
    data_stat_path: str = "../data/vlaw/meta_info/maniskill/stat.json"

    # ---- 训练 ----
    tag: str = "maniskill_wm"
    output_dir: str = "../checkpoints/vlaw/world_model"
    wandb_project_name: str = "vlaw_ctrl_world"
    train_batch_size: int = 1          # 单卡 4090 24GB
    gradient_accumulation_steps: int = 8   # 等效 batch=8
    mixed_precision: str = "fp16"
    learning_rate: float = 1e-5
    max_train_steps: int = 10000           # Phase-A 热身步数
    checkpointing_steps: int = 2000
    validation_steps: int = 500
    log_every_n_steps: int = 1   # mini-test 用 1, 正式训练用 10
    video_num: int = 2           # 验证视频数 (DROID=10, ManiSkill 数据少用 2)

    # ---- Phase 控制 ----
    # VLAW MODIFICATION: Phase A 仅训练 action_encoder + temporal attention
    freeze_unet_spatial: bool = True   # True=Phase-A; False=Phase-B (全量)

    # ---- 任务类型标记 ----
    # VLAW MODIFICATION: 供 train_wm.py 选择数据集类
    task_type: str = "maniskill"

    # ---- 验证 ----
    val_model_path: str = ckpt_path or ""
    decode_chunk_size: int = 4         # VAE 分块解码 (省显存)

    # VLAW MODIFICATION: 覆盖父类 __post_init__ 跳过 DROID task_type 校验
    def __post_init__(self) -> None:
        """ManiSkill 配置无需父类的 per-task 评估集配置, 直接跳过."""
        # 仅设置父类通用字段, 不触发 DROID 评估集逻辑
        self.gripper_max = self.gripper_max_dict.get(self.task_type, 0.75)
        self.task_name = "WM_maniskill"
        # val_dataset_dir 等 DROID 字段留空 (ManiSkill 推理使用 HDF5)
        self.val_dataset_dir = getattr(self, "val_dataset_dir", "../data/vlaw/demos")
        self.val_id = getattr(self, "val_id", [])
