# 01
# Train
CUDA_VISIBLE_DEVICES=4 python main_avvp.py --mode train \
  --audio_dir /home/jxl220096/data/llp/feats/vggish/ \
  --video_dir /home/jxl220096/data/llp/feats/res152/ \
  --st_dir /home/jxl220096/data/llp/feats/r2plus1d_18/ \
  --gpu 4 \
  --mamba_flag 'han_cmatt_to_mamba - hidden_state_simple' \
  --mamba_layers 1 \
  --cross True >./apr_output_logs/train_apr1_han_cmatt_to_mamba-hidden_state_simple.log

# 02
# Test
CUDA_VISIBLE_DEVICES=4 python main_avvp.py --mode test \
  --audio_dir /home/jxl220096/data/llp/feats/vggish/ \
  --video_dir /home/jxl220096/data/llp/feats/res152/ \
  --st_dir /home/jxl220096/data/llp/feats/r2plus1d_18/ \
  --gpu 4 \
  --mamba_flag 'han_cmatt_to_mamba - hidden_state_simple' \
  --mamba_layers 1 \
  --cross True >./apr_output_logs/test_apr1_han_cmatt_to_mamba-hidden_state_simple.log

# 03
CUDA_VISIBLE_DEVICES=2 python main_avvp.py --mode train \
  --audio_dir /home/jxl220096/data/llp/feats/vggish/ \
  --video_dir /home/jxl220096/data/llp/feats/res152/ \
  --st_dir /home/jxl220096/data/llp/feats/r2plus1d_18/ \
  --gpu 2 --lr 1e-4 \
  --mamba_flag 'han selfatt to mamba - hidden_state_simple' \
  --mamba_layers 1 \
  --cross True >./apr_output_logs/train_apr1_5_han_selfatt_to_mamba-hidden_state_simple.log

# 04
# Test
CUDA_VISIBLE_DEVICES=2 python main_avvp.py --mode test \
  --audio_dir /home/jxl220096/data/llp/feats/vggish/ \
  --video_dir /home/jxl220096/data/llp/feats/res152/ \
  --st_dir /home/jxl220096/data/llp/feats/r2plus1d_18/ \
  --gpu 2 --lr 1e-4 \
  --mamba_flag 'han selfatt to mamba - hidden_state_simple' \
  --mamba_layers 1 \
  --cross True >./apr_output_logs/test_apr1_5_han_selfatt_to_mamba-hidden_state_simple.log
