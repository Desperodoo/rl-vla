#\!/bin/bash
source /home/wjz/miniconda3/etc/profile.d/conda.sh
conda activate rlft_ms3
cd /home/wjz/rl-vla
CUDA_VISIBLE_DEVICES=3 nohup python rlft/vlaw/scripts/run_imagination.py --wm_ckpt checkpoints/vlaw/world_model/iter1_v3_ext/checkpoint-600.pt --num_trajs 20 --output_dir data/vlaw/synthetic/wm_eval_step600 --gpu_id 0 --visualize --vis_count 5 > logs/vlaw/imagination_eval_step600.log 2>&1 &
PID1=$\!
CUDA_VISIBLE_DEVICES=8 nohup python rlft/vlaw/scripts/run_imagination.py --wm_ckpt checkpoints/vlaw/world_model/iter1_v3_ext/checkpoint-1000.pt --num_trajs 20 --output_dir data/vlaw/synthetic/wm_eval_step1000 --gpu_id 0 --visualize --vis_count 5 > logs/vlaw/imagination_eval_step1000.log 2>&1 &
PID2=$\!
CUDA_VISIBLE_DEVICES=9 nohup python rlft/vlaw/scripts/run_imagination.py --wm_ckpt checkpoints/vlaw/world_model/iter1_v3_ext/checkpoint-1400.pt --num_trajs 20 --output_dir data/vlaw/synthetic/wm_eval_step1400 --gpu_id 0 --visualize --vis_count 5 > logs/vlaw/imagination_eval_step1400.log 2>&1 &
PID3=$\!
echo "$PID1 $PID2 $PID3" > /home/wjz/rl-vla/logs/vlaw/imagination_eval_pids.txt
echo "Job1=$PID1 Job2=$PID2 Job3=$PID3"
sleep 5
ps -p $PID1 -o pid,stat,cmd --no-headers 2>/dev/null && echo "PID $PID1: RUNNING" || echo "PID $PID1: NOT RUNNING"
ps -p $PID2 -o pid,stat,cmd --no-headers 2>/dev/null && echo "PID $PID2: RUNNING" || echo "PID $PID2: NOT RUNNING"
ps -p $PID3 -o pid,stat,cmd --no-headers 2>/dev/null && echo "PID $PID3: RUNNING" || echo "PID $PID3: NOT RUNNING"
ls -la logs/vlaw/imagination_eval_step600.log logs/vlaw/imagination_eval_step1000.log logs/vlaw/imagination_eval_step1400.log
