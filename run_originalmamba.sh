# export LD_LIBRARY_PATH=/people/cs/w/wxz220013/anaconda3/envs/OOD_backup/lib

# 01
# Train
CUDA_VISIBLE_DEVICES=5 python main_avvp.py --mode train \
  --audio_dir /data/jxl220096/dataset/llp/feats/vggish/ \
  --video_dir /data/jxl220096/dataset/llp/feats/res152/ \
  --st_dir /data/jxl220096/dataset/llp/feats/r2plus1d_18/ \
  --gpu 5 \
  --mamba_flag 'han cmatt to orimamba' \
  --mamba_layers 1 \
  --cross True >./apr_output_logs/train_apr2_han_cmatt_to_ori_mamba.log

# 02
# Test
CUDA_VISIBLE_DEVICES=5 python main_avvp.py --mode test \
  --audio_dir /data/jxl220096/dataset/llp/feats/vggish/ \
  --video_dir /data/jxl220096/dataset/llp/feats/res152/ \
  --st_dir /data/jxl220096/dataset/llp/feats/r2plus1d_18/ \
  --gpu 5 \
  --mamba_flag 'han cmatt to orimamba' \
  --mamba_layers 1 \
  --cross True >./apr_output_logs/test_apr2_han_cmatt_to_ori_mamba.log

# 03
# Train
CUDA_VISIBLE_DEVICES=5 python main_avvp.py --mode train \
  --audio_dir /data/jxl220096/dataset/llp/feats/vggish/ \
  --video_dir /data/jxl220096/dataset/llp/feats/res152/ \
  --st_dir /data/jxl220096/dataset/llp/feats/r2plus1d_18/ \
  --gpu 5 \
  --mamba_flag 'han selfatt to orimamba' \
  --mamba_layers 1 \
  --cross True >./apr_output_logs/train_apr2_han_selfatt_to_ori_mamba.log

# 04
# Test
CUDA_VISIBLE_DEVICES=5 python main_avvp.py --mode test \
  --audio_dir /data/jxl220096/dataset/llp/feats/vggish/ \
  --video_dir /data/jxl220096/dataset/llp/feats/res152/ \
  --st_dir /data/jxl220096/dataset/llp/feats/r2plus1d_18/ \
  --gpu 5 \
  --mamba_flag 'han selfatt to orimamba' \
  --mamba_layers 1 \
  --cross True >./apr_output_logs/test_apr2_han_selfatt_to_ori_mamba.log

# 05
# Train
CUDA_VISIBLE_DEVICES=5 python main_avvp.py --mode train \
  --audio_dir /data/jxl220096/dataset/llp/feats/vggish/ \
  --video_dir /data/jxl220096/dataset/llp/feats/res152/ \
  --st_dir /data/jxl220096/dataset/llp/feats/r2plus1d_18/ \
  --gpu 5 \
  --mamba_flag 'only_orimamba' \
  --mamba_layers 1 \
  --cross True >./apr_output_logs/train_apr2_han_to_ori_mamba.log

# 06
# Test
CUDA_VISIBLE_DEVICES=5 python main_avvp.py --mode test \
  --audio_dir /data/jxl220096/dataset/llp/feats/vggish/ \
  --video_dir /data/jxl220096/dataset/llp/feats/res152/ \
  --st_dir /data/jxl220096/dataset/llp/feats/r2plus1d_18/ \
  --gpu 5 \
  --mamba_flag 'only_orimamba' \
  --mamba_layers 1 \
  --cross True >./apr_output_logs/test_apr2_han_to_ori_mamba.log
