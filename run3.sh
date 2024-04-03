
# Train
CUDA_VISIBLE_DEVICES=2 python main_avvp.py --mode train \
  --audio_dir /home/jxl220096/data/llp/feats/vggish/ \
  --video_dir /home/jxl220096/data/llp/feats/res152/ \
  --st_dir /home/jxl220096/data/llp/feats/r2plus1d_18/ \
  --gpu 2 --lr 1e-4 \
  --mamba_flag 'han selfatt to mamba - hidden_state_simple' \
  --cross False >./apr_output_logs/train_apr1_5_han_selfatt_to_mamba-hidden_state_dynamic.log


# Test
CUDA_VISIBLE_DEVICES=2 python main_avvp.py --mode test \
  --audio_dir /home/jxl220096/data/llp/feats/vggish/ \
  --video_dir /home/jxl220096/data/llp/feats/res152/ \
  --st_dir /home/jxl220096/data/llp/feats/r2plus1d_18/ \
  --gpu 2 --lr 1e-4 \
  --mamba_flag 'han selfatt to mamba - hidden_state_simple' \
  --cross False >./apr_output_logs/test_apr1_5_han_selfatt_to_mamba-hidden_state_dynamic.log

  
# Train
# CUDA_VISIBLE_DEVICES=3 python main_avvp.py --mode train \
#   --audio_dir /home/jxl220096/data/llp/feats/vggish/ \
#   --video_dir /home/jxl220096/data/llp/feats/res152/ \
#   --st_dir /home/jxl220096/data/llp/feats/r2plus1d_18/ \
#   --gpu 3 \
#   --mamba_flag 'only_mamba' \
#   --cross False >./output_logs/train_mar31_5_han_to_mamba.log


# # Test
# CUDA_VISIBLE_DEVICES=3 python main_avvp.py --mode test \
#   --audio_dir /home/jxl220096/data/llp/feats/vggish/ \
#   --video_dir /home/jxl220096/data/llp/feats/res152/ \
#   --st_dir /home/jxl220096/data/llp/feats/r2plus1d_18/ \
#   --gpu 3 \
#   --mamba_flag 'only_mamba' \
#   --cross False >./output_logs/test_mar31_5_han_to_mamba.log
