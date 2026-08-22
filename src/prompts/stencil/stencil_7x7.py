import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, stride=1, padding=3)
    
    def forward(self, x):
        x = self.conv1(x)
        return x

# Test code
batch_size = 1

def get_inputs():
    # return [torch.rand(batch_size, 1, 10240, 10240)]
    return [torch.rand(batch_size, 1, 5120, 5120)]
    # return [torch.rand(batch_size, 1, 128, 128)]

def get_init_inputs():
    return []


if __name__ == "__main__":
    model = Model().cuda()
    compiled_model = torch.compile(model)

    y = compiled_model(torch.rand(1, 1, 5120, 5120).cuda())
