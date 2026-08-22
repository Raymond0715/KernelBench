import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import math

import pdb

def set_seed(seed: int):
    torch.manual_seed(seed)
    # NOTE: this only sets on current cuda device
    torch.cuda.manual_seed(seed)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, stride=1, padding=3)
    
    def forward(self, x):
        x = self.conv1(x)
        return x


def convstencil_kernel(pid, tensor_a_ptr, tensor_b_ptr_0,  tensor_b_ptr_1, bias,
                       output_ptr, ldm, COL, BLOCK_ROW, BLOCK_COL):
    # num_pid_col = math.ceil(COL/BLOCK_COL)
    num_pid_col = COL//BLOCK_COL

    pid_row = pid // num_pid_col
    pid_col = pid % num_pid_col

    block_base = pid_row * BLOCK_ROW * ldm + pid_col * BLOCK_COL
        
    # offsets for weight
    offsets_w_col = np.arange(0, 16)[None, :]
    offsets_w_row = np.arange(0, 64)[:, None]
    offsets_w = offsets_w_row * 16 + offsets_w_col

    # offsets for output
    offsets_o_col = np.arange(0, 16)[None, :]
    offsets_o_row = np.arange(0, 16)[:, None]


    for i in range(8):

        # offsets for input
        offsets_in_col = np.arange(0, 64)[None, :]
        offsets_in_row = np.arange(0, 16)[:, None]

        offsets_in = block_base + \
            offsets_in_col // 7 * ldm + offsets_in_row * 8 + offsets_in_col % 7

        # data
        mat_a_0 = tensor_a_ptr[offsets_in[:, 0:56]]
        mat_a_1 = tensor_a_ptr[offsets_in[:, 0:56] + 7]

        # kernel pointer
        mat_b_0 = tensor_b_ptr_0[offsets_w[0:56,:]]
        mat_b_1 = tensor_b_ptr_1[offsets_w[0:56,:]]

        output_0 = mat_a_0@mat_b_0
        output_1 = mat_a_1@mat_b_1
        output = output_0 + output_1 + bias

        output_ptr_tile = block_base + \
            offsets_o_col // 8 * ldm + offsets_o_col % 8 + offsets_o_row * 8

        output_ptr[output_ptr_tile.flatten()]=output.flatten()
        block_base += 2*ldm


def convstencil(grid, inputs, kernel0, kernel1, bias):
    output = torch.empty_like(inputs).numpy()
    output_1d = output.flatten()
    ROW, COL = output.shape[-2], output.shape[-1]
    inputs_np = inputs.numpy().flatten()
    for pid in range(grid):
        convstencil_kernel(
            pid, inputs_np, kernel0, kernel1, bias, output_1d, COL,
            COL=COL, BLOCK_ROW=16, BLOCK_COL=128)

    return output_1d


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.register_buffer("param_matrix", torch.zeros(2, 56, 16))
        for col in range(8):
            for i in range(7):
                for j in range(7):
                    if j >= col:
                        self.param_matrix[0][i*7+j][col] = self.conv1.weight[0][0][i][j-col]
                        self.param_matrix[0][i*7+j+7][col+8] = self.conv1.weight[0][0][i][j-col]


        for col in range(8):
            for i in range(7):
                for j in range(7):
                    if j < col:
                        self.param_matrix[1][i*7+j][col] = self.conv1.weight[0][0][i][j-col+7]
                        self.param_matrix[1][i*7+j+7][col+8] = self.conv1.weight[0][0][i][j-col+7]


    def forward(self, input_0):
        grid = math.ceil(input_0.shape[-2]/16)*math.ceil(input_0.shape[-1]/128)
        pad_h, pad_w = self.conv1.padding
        input_padding = F.pad(input_0, pad=(pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
        output = convstencil(
            grid, input_padding, self.param_matrix[0].detach().numpy().flatten(),
            self.param_matrix[1].detach().numpy().flatten(), self.conv1.bias.detach().numpy())
        return output


if __name__ == '__main__':
    inputs = torch.rand(1, 1, 512, 512)
    seed_num = 42

    set_seed(seed_num)
    model = Model()
    y = model(inputs)

    set_seed(seed_num)
    modelnew = ModelNew()
    y_new = modelnew(inputs)
    pdb.set_trace()
