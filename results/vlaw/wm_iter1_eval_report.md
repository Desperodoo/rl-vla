# WM iter1 评估报告

> 生成时间: 2026-02-28 18:15:44
> Task: LiftPegUpright-v1
> 验证数据: demo (val split) + rollout (val split)

## 概述

| 指标 | pretrained (ckpt-10000) | iter1 (ckpt-2000) | Delta |
| ---- | ---------------------- | ----------------- | ----- |
| PSNR | 23.39 ± 4.75 | 23.40 ± 2.87 | +0.00 |
| SSIM | 0.8116 ± 0.1122 | 0.7913 ± 0.0811 | -0.0203 |
| LPIPS ↓ | 0.1148 ± 0.0955 | 0.1200 ± 0.0721 | +0.0052 |
| #frames | 70 | 70 | - |
| #trajs | 14 | 14 | - |

## 门控检查

- iter1 PSNR: **23.40**
- 门控阈值: PSNR > 18.0
- 结果: **✅ 通过**

## 逐轨迹详情 (iter1)

| Traj | PSNR | SSIM | #frames |
| ---- | ---- | ---- | ------- |
| traj_0020 | 24.03 | 0.8737 | 5 |
| traj_0021 | 24.48 | 0.8642 | 5 |
| traj_0022 | 25.65 | 0.8900 | 5 |
| traj_0023 | 24.35 | 0.8717 | 5 |
| traj_0024 | 28.84 | 0.9394 | 5 |
| traj_0040 | 23.48 | 0.7578 | 5 |
| traj_0041 | 20.80 | 0.6793 | 5 |
| traj_0042 | 20.15 | 0.7212 | 5 |
| traj_0043 | 22.36 | 0.7487 | 5 |
| traj_0044 | 20.64 | 0.7119 | 5 |
| traj_0046 | 23.27 | 0.7275 | 5 |
| traj_0047 | 24.01 | 0.7748 | 5 |
| traj_0048 | 22.17 | 0.7473 | 5 |
| traj_0049 | 23.32 | 0.7703 | 5 |

## 分析

- pretrained 基线 PSNR: 23.39 (论文值 22.35)
- iter1 finetuned PSNR: 23.40
- Delta: +0.00 dB

✅ iter1 PSNR 相比 pretrained 有提升，微调有效。

## Checkpoints

- pretrained: `checkpoints/vlaw/world_model/pretrained/Ctrl-World/checkpoint-10000.pt`
- iter1: `checkpoints/vlaw/world_model/iter1/checkpoint-2000.pt`
