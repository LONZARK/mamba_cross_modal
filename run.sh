# export LD_LIBRARY_PATH=/people/cs/w/wxz220013/anaconda3/envs/OOD_backup/lib


# Train
CUDA_VISIBLE_DEVICES=3 python main_avvp.py --mode train \
  --audio_dir /home/jxl220096/data/llp/feats/vggish/ \
  --video_dir /home/jxl220096/data/llp/feats/res152/ \
  --st_dir /home/jxl220096/data/llp/feats/r2plus1d_18/ \
  --gpu 3 \
  --mamba_flag 'han cmatt to mamba' \
  --cross False >./apr_output_logs/train_apr1_han_cmatt_to_cross_mamba.log


# Test
CUDA_VISIBLE_DEVICES=3 python main_avvp.py --mode test \
  --audio_dir /home/jxl220096/data/llp/feats/vggish/ \
  --video_dir /home/jxl220096/data/llp/feats/res152/ \
  --st_dir /home/jxl220096/data/llp/feats/r2plus1d_18/ \
  --gpu 3 \
  --mamba_flag 'han cmatt to mamba' \
  --cross False >./apr_output_logs/test_apr1_han_cmatt_to_cross_mamba.log


# Train
CUDA_VISIBLE_DEVICES=3 python main_avvp.py --mode train \
  --audio_dir /home/jxl220096/data/llp/feats/vggish/ \
  --video_dir /home/jxl220096/data/llp/feats/res152/ \
  --st_dir /home/jxl220096/data/llp/feats/r2plus1d_18/ \
  --gpu 3 \
  --mamba_flag 'han selfatt to mamba' \
  --cross False >./apr_output_logs/train_apr1_han_selfatt_to_cross_mamba.log


# Test
CUDA_VISIBLE_DEVICES=3 python main_avvp.py --mode test \
  --audio_dir /home/jxl220096/data/llp/feats/vggish/ \
  --video_dir /home/jxl220096/data/llp/feats/res152/ \
  --st_dir /home/jxl220096/data/llp/feats/r2plus1d_18/ \
  --gpu 3 \
  --mamba_flag 'han selfatt to mamba' \
  --cross False >./apr_output_logs/test_apr1_han_selfatt_to_cross_mamba.log


