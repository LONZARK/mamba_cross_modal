# export LD_LIBRARY_PATH=/people/cs/w/wxz220013/anaconda3/envs/OOD_backup/lib


# Train
python main_avvp.py --mode train \
  --audio_dir /home/jxl220096/data/llp/feats/vggish/ \
  --video_dir /home/jxl220096/data/llp/feats/res152/ \
  --st_dir /home/jxl220096/data/llp/feats/r2plus1d_18/ \
  --gpu 0 \
  --mamba_flag 'han selfatt to mamba' \
  --cross False >./output_logs/train_mar29_HAN_selfatt_to_5orignal_mamba.log


# Test
python main_avvp.py --mode test \
  --audio_dir /home/jxl220096/data/llp/feats/vggish/ \
  --video_dir /home/jxl220096/data/llp/feats/res152/ \
  --st_dir /home/jxl220096/data/llp/feats/r2plus1d_18/ \
  --gpu 0 \
  --mamba_flag 'han selfatt to mamba' \
  --cross False >./output_logs/test_mar29_HAN_selfatt_to_5orignal_mamba.log


# Train
python main_avvp.py --mode train \
  --audio_dir /home/jxl220096/data/llp/feats/vggish/ \
  --video_dir /home/jxl220096/data/llp/feats/res152/ \
  --st_dir /home/jxl220096/data/llp/feats/r2plus1d_18/ \
  --gpu 0 \
  --mamba_flag 'han selfatt to mamba' \
  --cross True >./output_logs/train_mar29_HAN_selfatt_to_5crossmamba.log

# Test
python main_avvp.py --mode test \
  --audio_dir /home/jxl220096/data/llp/feats/vggish/ \
  --video_dir /home/jxl220096/data/llp/feats/res152/ \
  --st_dir /home/jxl220096/data/llp/feats/r2plus1d_18/ \
  --gpu 0 \
  --mamba_flag 'han selfatt to mamba' \
  --cross True >./output_logs/test_mar29_HAN_selfatt_to_5crossmamba.log




