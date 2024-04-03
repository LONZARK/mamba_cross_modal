# export LD_LIBRARY_PATH=/people/cs/w/wxz220013/anaconda3/envs/OOD_backup/lib

# 01
# Train
CUDA_VISIBLE_DEVICES=3 python main_avvp.py --mode train \
  --audio_dir /data/jxl220096/dataset/llp/feats/vggish/ \
  --video_dir /data/jxl220096/dataset/llp/feats/res152/ \
  --st_dir /data/jxl220096/dataset/llp/feats/r2plus1d_18/ \
  --gpu 3 \
  --mamba_flag 'han cmatt to crossmamba' \
  --mamba_layers 5 \
  --cross True >./apr_output_logs/train_apr2_han_cmatt_to_cross_mamba_5.log

# 02
# Test
CUDA_VISIBLE_DEVICES=3 python main_avvp.py --mode test \
  --audio_dir /data/jxl220096/dataset/llp/feats/vggish/ \
  --video_dir /data/jxl220096/dataset/llp/feats/res152/ \
  --st_dir /data/jxl220096/dataset/llp/feats/r2plus1d_18/ \
  --gpu 3 \
  --mamba_flag 'han cmatt to crossmamba' \
  --mamba_layers 5 \
  --cross True >./apr_output_logs/test_apr2_han_cmatt_to_cross_mamba_5.log

# 03
# Train
CUDA_VISIBLE_DEVICES=3 python main_avvp.py --mode train \
  --audio_dir /data/jxl220096/dataset/llp/feats/vggish/ \
  --video_dir /data/jxl220096/dataset/llp/feats/res152/ \
  --st_dir /data/jxl220096/dataset/llp/feats/r2plus1d_18/ \
  --gpu 3 \
  --mamba_flag 'han selfatt to crossmamba' \
  --mamba_layers 5 \
  --cross True >./apr_output_logs/train_apr2_han_selfatt_to_cross_mamba_5.log

# 04
# Test
CUDA_VISIBLE_DEVICES=3 python main_avvp.py --mode test \
  --audio_dir /data/jxl220096/dataset/llp/feats/vggish/ \
  --video_dir /data/jxl220096/dataset/llp/feats/res152/ \
  --st_dir /data/jxl220096/dataset/llp/feats/r2plus1d_18/ \
  --gpu 3 \
  --mamba_flag 'han selfatt to crossmamba' \
  --mamba_layers 5 \
  --cross True >./apr_output_logs/test_apr2_han_selfatt_to_cross_mamba_5.log

# 05
# Train
CUDA_VISIBLE_DEVICES=3 python main_avvp.py --mode train \
  --audio_dir /data/jxl220096/dataset/llp/feats/vggish/ \
  --video_dir /data/jxl220096/dataset/llp/feats/res152/ \
  --st_dir /data/jxl220096/dataset/llp/feats/r2plus1d_18/ \
  --gpu 3 \
  --mamba_flag 'only_crossmamba' \
  --mamba_layers 5 \
  --cross True >./apr_output_logs/train_apr2_han_to_cross_mamba_5.log

# 06
# Test
CUDA_VISIBLE_DEVICES=3 python main_avvp.py --mode test \
  --audio_dir /data/jxl220096/dataset/llp/feats/vggish/ \
  --video_dir /data/jxl220096/dataset/llp/feats/res152/ \
  --st_dir /data/jxl220096/dataset/llp/feats/r2plus1d_18/ \
  --gpu 3 \
  --mamba_flag 'only_crossmamba' \
  --mamba_layers 5 \
  --cross True >./apr_output_logs/test_apr2_han_to_cross_mamba_5.log

