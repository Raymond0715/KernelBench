# This is a hand-written convstencil operator based on Triton.
import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def convstencil_kernel(tensor_a_ptr, tensor_b_ptr_0,  tensor_b_ptr_1, bias, output_ptr, ldm,
                       ROW, COL, BLOCK_ROW: tl.constexpr, BLOCK_COL:tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_col = tl.cdiv(COL, BLOCK_COL)

    pid_row = pid // num_pid_col
    pid_col = pid % num_pid_col

        
    # offsets and mask for weight
    offsets_w_col = tl.arange(0, 16)[None, :]
    offsets_w_row = tl.arange(0, 64)[:, None]

    offsets_w = offsets_w_row * 16 + offsets_w_col

    mask_w_row = offsets_w_row < 56

    # offsets for output
    offsets_o_col = tl.arange(0, 16)[None, :]
    offsets_o_row = tl.arange(0, 16)[:, None]

    # block offsets
    block_base = pid_row * BLOCK_ROW * ldm + pid_col * BLOCK_COL
    output_ptr = output_ptr + block_base
    tensor_a_ptr_base = tensor_a_ptr + block_base

    # offsets for input
    offsets_in_col = tl.arange(0, 64)[None, :]
    offsets_in_row = tl.arange(0, 16)[:, None]

    offsets_in = offsets_in_col // 7 * ldm + offsets_in_row * 8 + offsets_in_col % 7
    mask_in = offsets_in_col < 56

    # data pointer
    mat_a_ptr_0 = tensor_a_ptr_base + offsets_in
    mat_a_ptr_1 = tensor_a_ptr_base + offsets_in + 7

    # kernel pointer
    mat_b_ptr_0 = tensor_b_ptr_0 + offsets_w
    mat_b_ptr_1 = tensor_b_ptr_1 + offsets_w

    mat_a_0 = tl.load(mat_a_ptr_0, mask=mask_in)
    mat_b_0 = tl.load(mat_b_ptr_0, mask=mask_w_row)
    output_0 = tl.dot(mat_a_0, mat_b_0)

    mat_a_0 = tl.load(mat_a_ptr_1, mask=mask_in)
    mat_b_1 = tl.load(mat_b_ptr_1, mask=mask_w_row)
    output_1 = tl.dot(mat_a_0, mat_b_1)

    bias_val = tl.load(bias)

    output = output_0 + output_1 + bias_val

    output_ptr_tile = output_ptr + offsets_o_col // 8 * ldm + offsets_o_col % 8 + offsets_o_row * 8

    tl.store(output_ptr_tile, output)


def convstencil(inputs, kernel0, kernel1, bias):
    output = torch.empty_like(inputs)
    ROW, COL = output.shape[-2], output.shape[-1]
    grid = lambda meta: (triton.cdiv(ROW, meta['BLOCK_ROW']) * triton.cdiv(COL, meta['BLOCK_COL']),)
    convstencil_kernel[grid](
            inputs, kernel0, kernel1, bias, output, COL,
            ROW=ROW, COL=COL, BLOCK_ROW=2, BLOCK_COL=128)

    return output


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
        row, col = input_0.shape[-2], input_0.shape[-1]
        pad_h, pad_w = self.conv1.padding
        input_padding = F.pad(input_0, pad=(pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
        output = convstencil(input_padding, self.param_matrix[0], self.param_matrix[1], self.conv1.bias)
        return output[:,:,:row,:col]


if __name__ == "__main__":
    batch_size = 1
    num_classes = 1000
    model = ModelNew().cuda()
    y = model(torch.rand(batch_size, 1, 5120, 5120).cuda())
