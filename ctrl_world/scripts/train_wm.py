# from diffusers import StableVideoDiffusionPipeline
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from models.ctrl_world import CrtlWorld

import numpy as np
import torch
import torch.nn as nn
import einops
from accelerate import Accelerator
import datetime
import os
from accelerate.logging import get_logger
from tqdm.auto import tqdm
import json
import shutil
import sys
from decord import VideoReader, cpu
import wandb
import swanlab
import mediapy
from models.ctrl_world import CrtlWorld
from config import wm_args
import math

# VLAW FIX: DeepSpeed 子进程可能丢失 conda PATH，显式设置 ffmpeg 路径
if shutil.which("ffmpeg") is None:
    _conda_prefix = sys.exec_prefix  # e.g. .../miniconda3/envs/ctrl_world
    _candidate = f"{_conda_prefix}/bin/ffmpeg"
    import os
    if os.path.isfile(_candidate):
        mediapy.set_ffmpeg(_candidate)


def main(args):
    logger = get_logger(__name__, log_level="INFO")
    # VLAW MODIFICATION: 禁用 SwanLab (BUG-008: SQLite 并发崩溃; 改用纯 wandb offline)
    # swanlab.sync_wandb() 会拦截 wandb.log() 并触发 SQLite commit 错误，完全跳过
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with='wandb',
        project_dir=args.output_dir
    )

    # model and optimizer
    model = CrtlWorld(args)
    if args.ckpt_path is not None:
        print(f"Loading checkpoint from {args.ckpt_path}!")
        state_dict = torch.load(args.ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
    model.to(accelerator.device)
    model.train()

    # VLAW MODIFICATION: 启用 gradient checkpointing 以降低显存使用
    model.unet.enable_gradient_checkpointing()

    # VLAW MODIFICATION: Phase-A / Phase-B 冻结策略
    # Phase-A (freeze_unet_spatial=True):  只训练 action_encoder + UNet temporal attention
    # Phase-B (freeze_unet_spatial=False): 解冻 UNet 全部, 全量微调
    if getattr(args, 'freeze_unet_spatial', False):
        # 冻结 UNet 全部参数, 然后解冻 temporal attention 层
        model.unet.requires_grad_(False)
        for name, module in model.unet.named_modules():
            if 'temporal' in name.lower() or 'temp_attn' in name.lower():
                module.requires_grad_(True)
        trainable_params = list(model.action_encoder.parameters()) + \
            [p for p in model.unet.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
        if accelerator.is_main_process:
            n_trainable = sum(p.numel() for p in trainable_params) / 1e6
            print(f"[Phase-A] 冻结 UNet 空间层; 可训练参数: {n_trainable:.2f}M")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        if accelerator.is_main_process:
            n_trainable = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"[Phase-B] 全量微调; 可训练参数: {n_trainable:.2f}M")

    # logs
    if accelerator.is_main_process:
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        tag = args.tag
        run_name = f"train_{now}_{tag}"
        accelerator.init_trackers(args.wandb_project_name,config={}, init_kwargs={"wandb":{"name":run_name}})
        os.makedirs(args.output_dir, exist_ok=True)
        # count parameters num in each part
        num_params = sum(p.numel() for p in model.unet.parameters())
        print(f"Number of parameters in the unet: {num_params/1000000:.2f}M")
        num_params = sum(p.numel() for p in model.vae.parameters())
        print(f"Number of parameters in the vae: {num_params/1000000:.2f}M")
        num_params = sum(p.numel() for p in model.image_encoder.parameters())
        print(f"Number of parameters in the image_encoder: {num_params/1000000:.2f}M")
        num_params = sum(p.numel() for p in model.text_encoder.parameters())
        print(f"Number of parameters in the text_encoder: {num_params/1000000:.2f}M")
        num_params = sum(p.numel() for p in model.action_encoder.parameters())
        print(f"Number of parameters in the action_encoder: {num_params/1000000:.2f}M")

    # train and val datasets
    # VLAW MODIFICATION: ManiSkill 数据集分支
    if getattr(args, 'task_type', 'droid') == 'maniskill':
        from dataset.dataset_maniskill import Dataset_ManiSkill
        train_dataset = Dataset_ManiSkill(args, mode='train')
        val_dataset = Dataset_ManiSkill(args, mode='val')
    else:
        from dataset.dataset_droid_exp33 import Dataset_mix
        train_dataset = Dataset_mix(args,mode='train')
        val_dataset = Dataset_mix(args,mode='val')
    # VLAW MODIFICATION: 增加 num_workers 和 pin_memory 以加速数据加载
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.train_batch_size,
        shuffle=args.shuffle,
        num_workers=getattr(args, 'num_workers', 4),
        pin_memory=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.train_batch_size,
        shuffle=args.shuffle,
        num_workers=getattr(args, 'num_workers', 4),
        pin_memory=True,
    )

    # Make all parameters contiguous (required for DeepSpeed broadcast)
    for p in model.parameters():
        if not p.data.is_contiguous():
            p.data = p.data.contiguous()

    # Prepare everything with our accelerator
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )
   
    ############################ training ##############################
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    _initial_step = getattr(args, 'initial_step', 0) or 0
    _remaining_steps = args.max_train_steps - _initial_step
    num_train_epochs = math.ceil(_remaining_steps * args.gradient_accumulation_steps*total_batch_size / len(train_dataloader))
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  checkpointing_steps = {args.checkpointing_steps}")
    logger.info(f"  validation_steps = {args.validation_steps}")
    global_step = getattr(args, 'initial_step', 0) or 0
    forward_step=0
    train_loss = 0.0
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    loss_gen, _ = model(batch)
                avg_loss = accelerator.gather(loss_gen.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item()/ args.gradient_accumulation_steps
                accelerator.backward(loss_gen)
                params_to_clip = model.parameters()
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                forward_step += 1
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                # log loss every N steps
                _log_every = getattr(args, 'log_every_n_steps', 100)
                if global_step % _log_every == 0:
                    progress_bar.set_postfix({"loss": train_loss})
                    accelerator.log({"train_loss": train_loss / _log_every}, step=global_step)
                    train_loss = 0.0
                # save ckpt every checkpointing_steps
                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pt")
                    torch.save(accelerator.unwrap_model(model).state_dict(), save_path)
                    logger.info(f"Saved checkpoint to {save_path}")
                # generate video every validation_steps
                if global_step % args.validation_steps == 5 and accelerator.is_main_process:
                    model.eval()
                    with accelerator.autocast():
                        for id in range(args.video_num):
                            validate_video_generation(model, val_dataset, args,global_step, args.output_dir, id, accelerator)
                    model.train()

                if global_step >= args.max_train_steps:
                    break
        if global_step >= args.max_train_steps:
            break


def main_val(args):
    accelerator = Accelerator()
    model = CrtlWorld(args)
    # load form val_model_path
    print("load from val_model_path",args.val_model_path)
    model.load_state_dict(torch.load(args.val_model_path))
    model.to(accelerator.device)
    model.eval()
    validate_video_generation(model, None, args, 0, 'output', 0, accelerator, load_from_dataset=False)
    
            

def validate_video_generation(model, val_dataset, args, train_steps, videos_dir, id, accelerator, load_from_dataset=True):
    device = accelerator.device
    pipeline = model.module.pipeline if accelerator.num_processes > 1 else model.pipeline
    videos_row = args.video_num if not args.debug else 1
    videos_col = 2

    # sample from val dataset
    # VLAW MODIFICATION: 防止 val_dataset 太小导致 range() step=0
    _step = max(1, int(len(val_dataset) / videos_row / videos_col))
    batch_id = list(range(0, len(val_dataset), _step))
    batch_id = batch_id[int(id*(videos_col)):int((id+1)*(videos_col))]
    if len(batch_id) == 0:
        print(f"[validate_video] 跳过 id={id}: val_dataset 太小 (len={len(val_dataset)})")
        return
    batch_list = [val_dataset.__getitem__(id) for id in batch_id]
    video_gt = torch.cat([t['latent'].unsqueeze(0) for i,t in enumerate(batch_list)],dim=0).to(device, non_blocking=True)
    text = [t['text'] for i,t in enumerate(batch_list)]
    actions = torch.cat([t['action'].unsqueeze(0) for i,t in enumerate(batch_list)],dim=0).to(device, non_blocking=True)
    his_latent_gt, future_latent_ft = video_gt[:,:args.num_history], video_gt[:,args.num_history:]
    current_latent = future_latent_ft[:,0]
    print("image",current_latent.shape, 'action', actions.shape)
    # VLAW MODIFICATION: Bug 1 修复 —— 根据 args.height/width 动态计算 latent shape，兼容 ManiSkill (48,24) 和 DROID (72,40)
    expected_lat_h = args.height // 8   # ManiSkill: 384//8=48, DROID: 576//8=72
    expected_lat_w = args.width // 8    # ManiSkill: 192//8=24, DROID: 320//8=40
    assert current_latent.shape[1:] == (4, expected_lat_h, expected_lat_w), \
        f"current_latent shape {current_latent.shape[1:]} != (4, {expected_lat_h}, {expected_lat_w})"
    assert actions.shape[1:] == (int(args.num_frames+args.num_history), args.action_dim)

    # start generate
    with torch.no_grad():
        bsz = actions.shape[0]
        # VLAW MODIFICATION: 确保 validation 时 action dtype 与模型参数一致 (DeepSpeed fp16)
        _ae = model.module.action_encoder if accelerator.num_processes > 1 else model.action_encoder
        _ae_dtype = next(_ae.parameters()).dtype
        actions = actions.to(dtype=_ae_dtype)
        action_latent = model.module.action_encoder(actions, text, model.module.tokenizer, model.module.text_encoder, args.frame_level_cond) if accelerator.num_processes > 1 else model.action_encoder(actions, text, model.tokenizer, model.text_encoder,args.frame_level_cond) # (8, 1, 1024)
        print("action_latent",action_latent.shape)

        # VLAW MODIFICATION: Bug 2 修复 —— ManiSkill args.height 已是拼接后总高度，无需再乘倍数；DROID 需乘以 3
        # VLAW MODIFICATION: 确保 pipeline 输入 dtype 一致 (DeepSpeed fp16)
        current_latent = current_latent.to(dtype=_ae_dtype)
        his_latent_gt = his_latent_gt.to(dtype=_ae_dtype)
        _, pred_latents = CtrlWorldDiffusionPipeline.__call__(
            pipeline,
            image=current_latent,
            text=action_latent,
            width=args.width,
            height=args.height if getattr(args, 'task_type', 'droid') == 'maniskill' else int(3 * args.height),
            num_frames=args.num_frames,
            history=his_latent_gt,
            num_inference_steps=args.num_inference_steps,
            decode_chunk_size=args.decode_chunk_size,
            max_guidance_scale=args.guidance_scale,
            fps=args.fps,
            motion_bucket_id=args.motion_bucket_id,
            mask=None,
            output_type='latent',
            return_dict=False,
            frame_level_cond=args.frame_level_cond,
            his_cond_zero=args.his_cond_zero,
        )
    
    # VLAW MODIFICATION: Bug 3 修复 —— 根据 task_type 选择相机数量 (ManiSkill=2, DROID=3)
    n_cams = 2 if getattr(args, 'task_type', 'droid') == 'maniskill' else 3
    pred_latents = einops.rearrange(pred_latents, 'b f c (m h) (n w) -> (b m n) f c h w', m=n_cams, n=1) # (B, 8, 4, 32,32)
    video_gt = torch.cat([his_latent_gt, future_latent_ft], dim=1) # (B, 8, 4, 32,32)
    video_gt = einops.rearrange(video_gt, 'b f c (m h) (n w) -> (b m n) f c h w', m=n_cams, n=1) # (B, 8, 4, 32,32)
    
    # decode latent
    # VLAW MODIFICATION: cast chunks to VAE dtype for DeepSpeed fp16 compatibility
    _vae_dtype = next(pipeline.vae.parameters()).dtype
    if video_gt.shape[2] != 3:  
        decoded_video = []
        bsz,frame_num = video_gt.shape[:2]
        video_gt = video_gt.flatten(0,1)
        decode_kwargs = {}
        for i in range(0,video_gt.shape[0],args.decode_chunk_size):
            chunk = video_gt[i:i+args.decode_chunk_size]/pipeline.vae.config.scaling_factor
            chunk = chunk.to(dtype=_vae_dtype)
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        video_gt = torch.cat(decoded_video,dim=0)
        video_gt = video_gt.reshape(bsz,frame_num,*video_gt.shape[1:])
        
        decoded_video = []
        bsz,frame_num = pred_latents.shape[:2]
        pred_latents = pred_latents.flatten(0,1)
        decode_kwargs = {}
        for i in range(0,pred_latents.shape[0],args.decode_chunk_size):
            chunk = pred_latents[i:i+args.decode_chunk_size]/pipeline.vae.config.scaling_factor
            chunk = chunk.to(dtype=_vae_dtype)
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        videos = torch.cat(decoded_video,dim=0)
        videos = videos.reshape(bsz,frame_num,*videos.shape[1:])

    video_gt = ((video_gt / 2.0 + 0.5).clamp(0, 1)*255)
    video_gt = video_gt.to(pipeline.unet.dtype).detach().cpu().numpy().transpose(0,1,3,4,2).astype(np.uint8)
    videos = ((videos / 2.0 + 0.5).clamp(0, 1)*255)
    videos = videos.to(pipeline.unet.dtype).detach().cpu().numpy().transpose(0,1,3,4,2).astype(np.uint8) #(2,16,256,256,3)
    videos = np.concatenate([video_gt[:, :args.num_history],videos],axis=1) #(2,16,512,256,3)
    videos = np.concatenate([video_gt,videos],axis=-3) #(2,16,512,256,3)
    videos = np.concatenate([video for video in videos],axis=-2).astype(np.uint8) # (16,512,256*batch,3)
    
    os.makedirs(f"{videos_dir}/samples", exist_ok=True)
    filename = f"{videos_dir}/samples/train_steps_{train_steps}_{id}.mp4"
    mediapy.write_video(filename, videos, fps=2)
    return 



if __name__ == "__main__":
    # reset parameters with command line
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--svd_model_path', type=str, default=None)
    parser.add_argument('--clip_model_path', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--dataset_root_path', type=str, default=None)
    parser.add_argument('--dataset_meta_info_path', type=str, default=None)
    parser.add_argument('--dataset_names', type=str, default=None)
    # VLAW MODIFICATION: 新增 ManiSkill 所需参数
    parser.add_argument('--task_type', type=str, default=None,
                        help="'droid' 或 'maniskill'")
    parser.add_argument('--freeze_unet_spatial', type=lambda x: x.lower() in ('true', '1', 'yes'),
                        default=None, help="Phase-A=True, Phase-B=False")
    parser.add_argument('--data_stat_path', type=str, default=None)
    parser.add_argument('--width', type=int, default=None)
    parser.add_argument('--height', type=int, default=None)
    parser.add_argument('--action_dim', type=int, default=None)
    parser.add_argument('--num_frames', type=int, default=None)
    parser.add_argument('--num_history', type=int, default=None)
    parser.add_argument('--train_batch_size', type=int, default=None)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None)
    parser.add_argument('--mixed_precision', type=str, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--max_train_steps', type=int, default=None)
    parser.add_argument('--initial_step', type=int, default=None,
                        help='Resume from this global step (for correct checkpoint naming)')
    parser.add_argument('--checkpointing_steps', type=int, default=None)
    parser.add_argument('--validation_steps', type=int, default=None)
    parser.add_argument('--decode_chunk_size', type=int, default=None)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--log_every_n_steps', type=int, default=None,
                        help='Log train_loss every N optimizer steps (default: 100 DROID / 1 maniskill)')
    parser.add_argument('--video_num', type=int, default=None,
                        help='Number of validation videos to generate')
    args_new = parser.parse_args()

    # VLAW MODIFICATION: 根据 task_type 选择默认配置
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import wm_args, wm_args_maniskill
    if getattr(args_new, 'task_type', None) == 'maniskill':
        args = wm_args_maniskill()
    else:
        args = wm_args()

    def merge_args(args, new_args):
        for k, v in new_args.__dict__.items():
            if v is not None:
                args.__dict__[k] = v
        return args

    args = merge_args(args, args_new)

    main(args)

    # CUDA_VISIBLE_DEVICES=0,1 WANDB_MODE=offline accelerate launch --main_process_port 29501 train_wm.py --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info
    # CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 29506 unit_test2.py

    # args = Args()
    # from video_dataset.dataset_droid_exp33 import Dataset_mix
    # dataset = Dataset_mix(args,mode='val')
    # from torch.utils.data import DataLoader
    # dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=2)
    # model = CrtlWorld(args).to('cuda')
    # # print model parameter num
    # num_params = sum(p.numel() for p in model.parameters())
    # print(f"Number of parameters in the model: {num_params/1000000:.2f}M")
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-6)
    # total_elements = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
    # print(f"Total number of learnable parameters: {total_elements}")
    # model.train()
    

    # for batch in dataloader:
    #     print(batch['latent'].shape)
    #     print(batch['text'])
    #     print(batch['action'].shape)

    #     loss,_ = model(batch)
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()
    #     print(loss.item())





    # device = 'cuda'
    # video_encoder = VideoEncoder(hidden_size=1024).to(device)
    # # count the parameters of the model
    # num_params = sum(p.numel() for p in video_encoder.parameters())
    # print(f"Number of parameters in the model: {num_params/1000000:.2f}M")
    # vae_latent = torch.randn(8, 1, 4, 32, 32).to(device)
    # clip_latent = torch.randn(8, 20, 512).to(device)
    # current_img = video_encoder(vae_latent, clip_latent)
    # print(current_img.shape)  # (8, 1, 4, 32, 32)


    # pos_emb = get_2d_sincos_pos_embed(1024, 16)
    # print(pos_emb.shape)  # (256, 1024)
    # clip_emb = get_1d_sincos_pos_embed_from_grid(1024, np.arange(20))
    # print(clip_emb.shape)  # (20, 512)
