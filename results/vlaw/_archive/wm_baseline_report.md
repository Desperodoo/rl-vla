# WM Horizon Baseline Report

> Generated: 2026-02-26 22:59:22
> Task: LiftPegUpright-v1
> Data: data/vlaw/encoded/demos/LiftPegUpright-v1/LiftPegUpright-v1_demo_1771951465.h5 + data/vlaw/encoded/rollouts/iter1/LiftPegUpright-v1/LiftPegUpright-v1_real_1772017887.h5

## 概述

本报告评估 Ctrl-World 世界模型在 LiftPegUpright 数据上的视频预测质量，
按不同 horizon (5/10/15/20 帧) 分解指标，建立后续迭代对照基准。

## Checkpoints

- **pretrained**: `checkpoints/vlaw/world_model/pretrained/Ctrl-World/checkpoint-10000.pt` (25 trajs)
- **phase_a_step12000**: `checkpoints/vlaw/world_model/phase_a/checkpoint-12000.pt` (25 trajs)

## Horizon 分解对比

| Horizon | pretrained PSNR | phase_a_step12000 PSNR | pretrained SSIM | phase_a_step12000 SSIM | pretrained LPIPS | phase_a_step12000 LPIPS |
| ------- | ------ | ------ | ------ | ------ | ------ | ------ |
| 5 | 22.10 | 22.60 | 0.7946 | 0.6373 | 0.1371 | 0.1593 |
| 10 | 22.50 | 22.37 | 0.7976 | 0.6063 | 0.1327 | 0.1682 |
| 15 | 22.29 | 21.99 | 0.7917 | 0.5877 | 0.1369 | 0.1793 |
| 20 | 22.35 | 21.70 | 0.7943 | 0.5754 | 0.1329 | 0.1863 |

## 逐帧衰减曲线

| Frame | pretrained PSNR | pretrained SSIM | phase_a_step12000 PSNR | phase_a_step12000 SSIM |
| ----- | ------ | ------ | ------ | ------ |
| 1 | 29.01 | 0.9161 | 28.12 | 0.7019 |
| 2 | 22.39 | 0.8447 | 22.36 | 0.6288 |
| 3 | 21.73 | 0.8360 | 21.38 | 0.6227 |
| 4 | 19.24 | 0.7152 | 20.78 | 0.6177 |
| 5 | 18.11 | 0.6610 | 20.38 | 0.6151 |
| 6 | 30.22 | 0.9168 | 25.73 | 0.6448 |
| 7 | 23.92 | 0.8693 | 21.59 | 0.5487 |
| 8 | 21.66 | 0.8495 | 20.51 | 0.5296 |
| 9 | 16.99 | 0.6256 | 20.03 | 0.5162 |
| 10 | 16.25 | 0.5900 | 19.59 | 0.5046 |
| 11 | 29.53 | 0.9031 | 22.76 | 0.5552 |
| 12 | 21.69 | 0.8378 | 20.50 | 0.5135 |
| 13 | 20.42 | 0.8069 | 19.62 | 0.4921 |
| 14 | 16.47 | 0.6181 | 19.06 | 0.4803 |
| 15 | 15.41 | 0.5847 | 18.87 | 0.4789 |
| 16 | 31.63 | 0.9174 | 21.70 | 0.5366 |
| 17 | 22.32 | 0.8426 | 20.17 | 0.5001 |
| 18 | 21.02 | 0.8157 | 19.44 | 0.4845 |
| 19 | 19.44 | 0.7482 | 19.13 | 0.4820 |
| 20 | 19.23 | 0.7325 | 18.81 | 0.4760 |

## 对比分析

### phase_a_step12000 vs pretrained (Delta)

| Horizon | ΔPSNR | ΔSSIM | ΔLPIPS |
| ------- | ----- | ----- | ------ |
| 5 | +0.51 | -0.1573 | +0.0221 |
| 10 | -0.14 | -0.1913 | +0.0355 |
| 15 | -0.31 | -0.2040 | +0.0424 |
| 20 | -0.65 | -0.2189 | +0.0533 |

## 可视化

GT vs Predicted 对比图保存在: `results/vlaw/wm_baseline/`

### pretrained
- `pretrained/vis_traj_0000.png`
- `pretrained/vis_traj_0001.png`
- `pretrained/vis_traj_0002.png`
- `pretrained/vis_traj_0003.png`
- `pretrained/vis_traj_0004.png`

### phase_a_step12000
- `phase_a_step12000/vis_traj_0000.png`
- `phase_a_step12000/vis_traj_0001.png`
- `phase_a_step12000/vis_traj_0002.png`
- `phase_a_step12000/vis_traj_0003.png`
- `phase_a_step12000/vis_traj_0004.png`

## 结论

- PSNR > 18 为通过标准 (P2.3)
- **pretrained**: horizon-5 PSNR=22.10, horizon-20 PSNR=22.35, 衰减=-0.26dB
- **phase_a_step12000**: horizon-5 PSNR=22.60, horizon-20 PSNR=21.70, 衰减=0.90dB
