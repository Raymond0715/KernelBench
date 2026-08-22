import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.register_buffer("param_matrix", torch.zeros(2, 49, 8))
        for col in range(8):
            for i in range(7):
                for j in range(7):
                    if j >= col:
                        self.param_matrix[0][i*7+j][col] = self.conv1.weight[0][0][i][j-col]

        for col in range(8):
            for i in range(7):
                for j in range(7):
                    if j < col:
                        self.param_matrix[1][i*7+j][col] = self.conv1.weight[0][0][i][j-col+7]
    

    def forward(self, x):
        _, _, OY, OX = x.shape
        out = torch.zeros_like(x)
        pad_h, pad_w = self.conv1.padding
        x = F.pad(x, pad=(pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
        for input_row_idx in range(OY):
            for input_col_idx in range(0, OX, 64):
                mm_a_0 = torch.stack(
                        [x[0, 0,
                           input_row_idx:input_row_idx+7,
                           input_col_idx+8*i:input_col_idx+8*i+7
                           ].flatten()
                         for i in range(8)])

                mm_a_1 = torch.stack(
                        [x[0, 0,
                           input_row_idx:input_row_idx+7,
                           input_col_idx+8*i+7:input_col_idx+8*i+14
                           ].flatten()
                         for i in range(8)])

                acc_0 = torch.matmul(mm_a_0, self.param_matrix[0]).flatten()
                acc_1 = torch.matmul(mm_a_1, self.param_matrix[1]).flatten() + acc_0

                out[0, 0, input_row_idx, input_col_idx:input_col_idx+64] = acc_1 + self.conv1.bias[0]


        return out

# Test code
batch_size = 1

def get_inputs():
    # return [torch.rand(batch_size, 1, 10240, 10240)]
    return [torch.rand(batch_size, 1, 128, 128)]

def get_init_inputs():
    return []
#
#
# if __name__ == "__main__":
#     batch_size = 1
#     num_classes = 1000
#     model = ModelNew()
#     y = model(torch.rand(batch_size, 1, 128, 128))
