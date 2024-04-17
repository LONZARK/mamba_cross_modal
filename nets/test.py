
import numpy as np

# 假设 tensor1 和 tensor2 是形状为 (10, 512) 的两个 NumPy 数组
import torch
tensor1 = torch.randn([2, 3, 5])  # 第一个张量
tensor2 = torch.randn([2, 3, 5])  # 第二个张量



def expand(tensor1, tensor2):
    # Add a new axis/dimension in the appropriate position
    nor = torch.einsum('i j k,i l k->i j l k',tensor1, tensor2)
    tensor1_expanded = tensor1.unsqueeze(2)  # Adds a dimension at position 2
    tensor2_expanded = tensor2.unsqueeze(1)  # Adds a dimension at position 1

    # result_tensor = 0.5*tensor1_expanded + 0.5*tensor2_expanded
    result_tensor = tensor1_expanded * tensor2_expanded
    print(nor)
    print(result_tensor)
    return result_tensor

# result_tensor = expand(tensor1, tensor2)


def batch_mamba_scan(tensor, i, j):
    # Assume tensor shape is [batch_size, height, width, depth]
    batch_size, height, width, depth = tensor.size()

    element_in_second_dim = tensor[:, i, :, :]
    element_in_third_dim = tensor[:, :, j, :]

    element_left = element_in_second_dim[:, :j+1, :]
    element_right = element_in_second_dim[:, j:, :]
    element_right = torch.flip(element_right, dims=[1])
    
    element_up = element_in_third_dim[:, :i+1, :]
    element_down = element_in_third_dim[:, i:, :]
    element_down = torch.flip(element_down, dims=[1])
    
    return element_left, element_right, element_up, element_down


batch_size = 16
tensor = torch.randn(batch_size, 10, 10, 512)
batch_mamba_scan(tensor, 1, 2)




