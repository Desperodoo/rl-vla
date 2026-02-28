# WM iter1 评估报告

> 生成时间: 2026-02-28 15:19:45
> Task: LiftPegUpright-v1
> 验证数据: demo (val split) + rollout (val split)

## 概述

| 指标 | pretrained (ckpt-10000) | iter1 (ckpt-2000) | Delta |
| ---- | ---------------------- | ----------------- | ----- |
| PSNR | 23.01 ± 5.05 | 23.34 ± 2.72 | +0.33 |
| SSIM | 0.8014 ± 0.1182 | 0.7929 ± 0.0770 | -0.0085 |
| LPIPS ↓ | 0.1297 ± 0.1127 | 0.1190 ± 0.0697 | -0.0107 |
| #frames | 70 | 70 | - |
| #trajs | 14 | 14 | - |

## 门控检查

- iter1 PSNR: **23.34**
- 门控阈值: PSNR > 18.0
- 结果: **✅ 通过**

## 逐轨迹详情 (iter1)

| Traj | PSNR | SSIM | #frames |
| ---- | ---- | ---- | ------- |
| traj_0020 | 23.69 | 0.8687 | 5 |
| traj_0021 | 24.08 | 0.8606 | 5 |
| traj_0022 | 25.80 | 0.8908 | 5 |
| traj_0023 | 24.35 | 0.8695 | 5 |
| traj_0024 | 28.20 | 0.9347 | 5 |
| traj_0040 | 23.46 | 0.7628 | 5 |
| traj_0041 | 21.21 | 0.6926 | 5 |
| traj_0042 | 20.21 | 0.7284 | 5 |
| traj_0043 | 22.35 | 0.7579 | 5 |
| traj_0044 | 21.01 | 0.7148 | 5 |
| traj_0046 | 23.31 | 0.7393 | 5 |
| traj_0047 | 24.00 | 0.7715 | 5 |
| traj_0048 | 21.80 | 0.7433 | 5 |
| traj_0049 | 23.32 | 0.7651 | 5 |

## 分析

- pretrained 基线 PSNR: 23.01 (论文值 22.35)
- iter1 finetuned PSNR: 23.34
- Delta: +0.33 dB

✅ iter1 PSNR 相比 pretrained 有提升，微调有效。

## Checkpoints

- pretrained: `checkpoints/vlaw/world_model/pretrained/Ctrl-World/checkpoint-10000.pt`
- iter1: `checkpoints/vlaw/world_model/iter1/checkpoint-2000.pt`
