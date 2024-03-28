
import torch
# from mamba.mamba_ssm.modules.mamba_simple import Mamba

# batch, length, dim = 2, 64, 16
# x = torch.randn(batch, length, dim).to("cuda")
# model = Mamba(
#     # This module uses roughly 3 * expand * d_model^2 parameters
#     d_model=dim, # Model dimension d_model
#     d_state=16,  # SSM state expansion factor
#     d_conv=4,    # Local convolution width
#     expand=2,    # Block expansion factor
# ).to("cuda")
# y = model(x)
# assert y.shape == x.shape


import numpy as np
import matplotlib.pyplot as plt

data = np.load('/home/jxl220096/data/llp/feats/vggish/BjCEufrlXm4.npy')

# # Now `data` is a numpy array containing the data from the .npy file
# print(data)
# print(data.shape)

# print(len(data))
# print(len(data[0]))


# # Replace 'path_to_file.npy' with the path to your .npy file
# data = np.load('/data/wxz220013/llp/feats/res152/BjCEufrlXm4.npy')

# Now `data` is a numpy array containing the data from the .npy file
print(data)
print(data.shape)

print(len(data))
print(len(data[0]))

exit()

features = data  # Example features, replace this with your actual features

differences = np.linalg.norm(np.diff(features, axis=0), axis=1)

threshold = np.mean(differences) + np.std(differences)  

# Identify time steps where the change exceeds the threshold
sudden_changes = np.where(differences > threshold)[0] + 1  # +1 to account for the shift due to np.diff

print("Time steps with sudden changes:", sudden_changes)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(differences, label='Difference between consecutive frames')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.scatter(sudden_changes, differences[sudden_changes], color='red', label='Sudden changes')
plt.xlabel('Frame')
plt.ylabel('Difference')
plt.title('Identifying Sudden Changes in Video Visual Features')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('/people/cs/w/wxz220013/AVVP-ECCV20/data/LLP_dataset/Difference_in_Audio_Features.png')



exit()




# 初始化一个空字典来存储解析的数据
parsed_data = {}

# 定义日志文件的路径
log_file_path = '/people/cs/w/wxz220013/AVVP-ECCV20/output_logs/train_test_mar13_epoch200.log'

# 使用with语句打开文件，这样可以确保文件在读取后自动关闭
with open(log_file_path, 'r') as file:
    log_content = file.read()

# 按行分割日志内容
lines = log_content.split('\n')

# 遍历每行来提取信息
for line in lines:
    if line.startswith("Train Epoch:"):
        # 提取epoch编号
        epoch = int(line.split(":")[1].split()[0])
        loss_value = float(line.split(":")[2])
        parsed_data[epoch] = {}
        parsed_data[epoch]['loss'] = loss_value
    elif "F1" in line:
        # 提取F1分数及其描述
        parts = line.split(":")
        key = parts[0].strip()
        value = float(parts[1].strip())
        parsed_data[epoch][key] = value

# print(parsed_data)


import matplotlib.pyplot as plt

# Assuming the dictionary data is complete from epoch 1 to 200, this example only uses the final epoch's data for illustration.
# In practice, you should fill the data dictionary with actual values for all epochs.

# Initialize lists to store the data for plotting
epochs = list(range(1, 201))  # Placeholder for actual epoch numbers
loss_values = [parsed_data[epoch]['loss'] for epoch in epochs if epoch in parsed_data]
audio_segement_f1 = [parsed_data[epoch]['Audio Event Detection Segment-level F1'] for epoch in epochs if epoch in parsed_data]
visual_segement_f1 = [parsed_data[epoch]['Visual Event Detection Segment-level F1'] for epoch in epochs if epoch in parsed_data]
av_segement_f1 = [parsed_data[epoch]['Audio-Visual Event Detection Segment-level F1'] for epoch in epochs if epoch in parsed_data]

audio_Event_f1 = [parsed_data[epoch]['Audio Event Detection Event-level F1'] for epoch in epochs if epoch in parsed_data]
visual_Event_f1 = [parsed_data[epoch]['Visual Event Detection Event-level F1'] for epoch in epochs if epoch in parsed_data]
av_Event_f1 = [parsed_data[epoch]['Audio-Visual Event Detection Event-level F1'] for epoch in epochs if epoch in parsed_data]

# Plot the trends of loss and F1 scores over epochs
plt.figure(figsize=(10, 6))

plt.plot(epochs, loss_values, label='Loss', marker='o')
plt.plot(epochs, audio_segement_f1, label='Audio Event Detection Segment-level F1', marker='x')
plt.plot(epochs, visual_segement_f1, label='Visual Event Detection Segment-level F1', marker='x')
plt.plot(epochs, av_segement_f1, label='Audio-Visual Event Detection Segment-level F1', marker='x')
plt.plot(epochs, audio_Event_f1, label='Audio Event Detection Event-level F1', marker='^')
plt.plot(epochs, visual_Event_f1, label='Visual Event Detection Event-level F1', marker='^')
plt.plot(epochs, av_Event_f1, label='Audio-Visual Event Detection Event-level F1', marker='^')



plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Loss and F1 Score Trends over Epochs')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('/people/cs/w/wxz220013/AVVP-ECCV20/output_logs/test.png')
















exit()
import torch
print(torch.cuda.is_available())


# Testing PyKeops installation
import pykeops

# Changing verbose and mode
pykeops.verbose = True
pykeops.build_type = 'Debug'

# Clean up the already compiled files
pykeops.clean_pykeops()

# Test Numpy integration
pykeops.test_numpy_bindings()