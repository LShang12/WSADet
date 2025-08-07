#Credits to: https://github.com/detkov/Convolution-From-Scratch/
import torch
import numpy as np
from typing import List, Tuple, Union


# 在 convolution.py 中确保 calc_out_dims 接收正确的参数类型
def calc_out_dims(matrix, kernel_side, stride, dilation, padding):
    # 检查参数类型
    assert isinstance(padding, (tuple, list)), "padding 必须是元组或列表"
    assert isinstance(dilation, (tuple, list)), "dilation 必须是元组或列表"

    batch_size, n_channels, n, m = matrix.shape
    h_out = int((n + 2 * padding[0] - dilation[0] * (kernel_side - 1) - 1) // stride[0] + 1)
    w_out = int((m + 2 * padding[1] - dilation[1] * (kernel_side - 1) - 1) // stride[1] + 1)
    return h_out, w_out, batch_size, n_channels


def multiple_convs_kan_conv2d(
        matrix: torch.Tensor,
        kernels: list,  # 确保接收列表类型
        kernel_side: int,
        out_channels: int,
        stride: tuple = (1, 1),
        dilation: tuple = (1, 1),
        padding: tuple = (0, 0),
        device: str = "cuda"
) -> torch.Tensor:
    """支持单核和多核的卷积实现"""
    # 计算输出维度
    h_out, w_out, batch_size, n_channels = calc_out_dims(matrix, kernel_side, stride, dilation, padding)

    # 初始化输出张量
    matrix_out = torch.zeros((batch_size, out_channels, h_out, w_out)).to(device)

    # 特征展开
    unfold = torch.nn.Unfold((kernel_side, kernel_side),
                             dilation=dilation,
                             padding=padding,
                             stride=stride)
    unfolded = unfold(matrix).view(batch_size, n_channels, -1, h_out * w_out)

    # 处理每个输出通道
    for c_out in range(out_channels):
        if c_out < len(kernels):  # 安全处理核数量不足的情况
            kernel = kernels[c_out]
            # 应用单个KANLinear核
            conv_result = kernel(unfolded[:, c_out % n_channels, :, :].flatten(0, 1))
            conv_result = conv_result.view(batch_size, h_out, w_out)
            matrix_out[:, c_out, :, :] = conv_result

    return matrix_out


def add_padding(matrix: np.ndarray, 
                padding: Tuple[int, int]) -> np.ndarray:
    """Adds padding to the matrix. 

    Args:
        matrix (np.ndarray): Matrix that needs to be padded. Type is List[List[float]] casted to np.ndarray.
        padding (Tuple[int, int]): Tuple with number of rows and columns to be padded. With the `(r, c)` padding we addding `r` rows to the top and bottom and `c` columns to the left and to the right of the matrix

    Returns:
        np.ndarray: Padded matrix with shape `n + 2 * r, m + 2 * c`.
    """
    n, m = matrix.shape
    r, c = padding
    
    padded_matrix = np.zeros((n + r * 2, m + c * 2))
    padded_matrix[r : n + r, c : m + c] = matrix
    
    return padded_matrix
