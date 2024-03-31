# Train
python main_avvp.py --mode train \
  --audio_dir /home/jxl220096/data/llp/feats/vggish/ \
  --video_dir /home/jxl220096/data/llp/feats/res152/ \
  --st_dir /home/jxl220096/data/llp/feats/r2plus1d_18/ \
  --gpu 1 \
  --mamba_flag 'han cmatt to mamba' \
  --cross False >./output_logs/train_mar29_HAN_cmatt_to_5orignal_mamba.log

# Test
python main_avvp.py --mode test \
  --audio_dir /home/jxl220096/data/llp/feats/vggish/ \
  --video_dir /home/jxl220096/data/llp/feats/res152/ \
  --st_dir /home/jxl220096/data/llp/feats/r2plus1d_18/ \
  --gpu 1 \
  --mamba_flag 'han cmatt to mamba' \
  --cross False >./output_logs/test_mar29_HAN_cmatt_to_5orignal_mamba.log



# Train
python main_avvp.py --mode train \
  --audio_dir /home/jxl220096/data/llp/feats/vggish/ \
  --video_dir /home/jxl220096/data/llp/feats/res152/ \
  --st_dir /home/jxl220096/data/llp/feats/r2plus1d_18/ \
  --gpu 1 \
  --mamba_flag 'han cmatt to mamba' \
  --cross True >./output_logs/train_mar29_HAN_cmatt_to_5crossmamba.log

# Test
python main_avvp.py --mode test \
  --audio_dir /home/jxl220096/data/llp/feats/vggish/ \
  --video_dir /home/jxl220096/data/llp/feats/res152/ \
  --st_dir /home/jxl220096/data/llp/feats/r2plus1d_18/ \
  --gpu 1 \
  --mamba_flag 'han cmatt to mamba' \
  --cross True >./output_logs/test_mar29_HAN_cmatt_to_5crossmamba.log
