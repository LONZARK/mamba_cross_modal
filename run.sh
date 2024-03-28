# export LD_LIBRARY_PATH=/people/cs/w/wxz220013/anaconda3/envs/OOD_backup/lib


# Train
python main_avvp.py --mode train --audio_dir /home/jxl220096/data/llp/feats/vggish/ --video_dir /home/jxl220096/data/llp/feats/res152/ --st_dir /home/jxl220096/data/llp/feats/r2plus1d_18/ --gpu 1

# Test
python main_avvp.py --mode test --audio_dir /home/jxl220096/data/llp/feats/vggish/ --video_dir /home/jxl220096/data/llp/feats/res152/ --st_dir /home/jxl220096/data/llp/feats/r2plus1d_18/
