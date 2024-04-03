# 05
# Train
CUDA_VISIBLE_DEVICES=4 python main_avvp.py --mode train \
  --audio_dir /data/jxl220096/dataset/llp/feats/vggish/ \
  --video_dir /data/jxl220096/dataset/llp/feats/res152/ \
  --st_dir /data/jxl220096/dataset/llp/feats/r2plus1d_18/ \
  --gpu 4 \
  --mamba_flag 'han_cmatt_to_mamba - hidden_state_dynamic' \
  --mamba_layers 1 \
  --cross True >./apr_output_logs/train_apr1_han_cmatt_to_mamba-hidden_state_dynamic.log

# 06
# Test
CUDA_VISIBLE_DEVICES=4 python main_avvp.py --mode test \
  --audio_dir /data/jxl220096/dataset/llp/feats/vggish/ \
  --video_dir /data/jxl220096/dataset/llp/feats/res152/ \
  --st_dir /data/jxl220096/dataset/llp/feats/r2plus1d_18/ \
  --gpu 4 \
  --mamba_flag 'han_cmatt_to_mamba - hidden_state_dynamic' \
  --mamba_layers 1 \
  --cross True >./apr_output_logs/test_apr1_han_cmatt_to_mamba-hidden_state_dynamic.log

# 07
# Train
CUDA_VISIBLE_DEVICES=4 python main_avvp.py --mode train \
  --audio_dir /data/jxl220096/dataset/llp/feats/vggish/ \
  --video_dir /data/jxl220096/dataset/llp/feats/res152/ \
  --st_dir /data/jxl220096/dataset/llp/feats/r2plus1d_18/ \
  --gpu 4 \
  --mamba_flag 'han selfatt to mamba - hidden_state_dynamic' \
  --mamba_layers 1 \
  --cross True >./apr_output_logs/train_apr1_han_selfatt_to_mamba-hidden_state_dynamic.log

# 08
# Test
CUDA_VISIBLE_DEVICES=4 python main_avvp.py --mode test \
  --audio_dir /data/jxl220096/dataset/llp/feats/vggish/ \
  --video_dir /data/jxl220096/dataset/llp/feats/res152/ \
  --st_dir /data/jxl220096/dataset/llp/feats/r2plus1d_18/ \
  --gpu 4 \
  --mamba_flag 'han selfatt to mamba - hidden_state_dynamic' \
  --mamba_layers 1 \
  --cross True >./apr_output_logs/test_apr1_han_selfatt_to_mamba-hidden_state_dynamic.log