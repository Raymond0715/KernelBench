import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx

# class Model(nn.Module):
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, stride=1, padding=3)
    
    def forward(self, x):
        y = torch.zeros_like(x)

        pad_h, pad_w = self.conv1.padding
        x = F.pad(x, pad=(pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)

        KX = self.conv1.weight.shape[3]
        KY = self.conv1.weight.shape[2]
        Y = x.shape[2]
        X = x.shape[3]
        w_frag = torch.zeros(KY+Y, KX+X, Y, X)
        in_frag = torch.zeros(int(Y/2), int(X/8), KY+Y, KX+X)
        y_frag = torch.zeros(int(Y/2), int(X/8), 8, 8)


        for yt in range(0, int(Y/2)):
            for xt in range(0, int(X/8)):
                for yi in range(0, 2, 1):
                    for xi in range(0, 8, 1):
                        for p in range(yi, KY+yi, 1):
                            for q in range(xi, KX+xi, 1):
                                in_frag[yt][xt][p][q] = x[yt*2+p][xt*8+q]
                                w_frag[p][q][yi][xi] = self.conv1.weight[0][0][p-yi][q-xi]
                                y_frag[yt][xt][yi][xi]+=in_frag[yt][xt][p][q]*w_frag[p][q][yi][xi]
                                y[yt*2+yi][xt*8+xi] = y_frag[yt][xt][yi][xi]


        return y

# Test code
batch_size = 1

def get_inputs():
    return [torch.rand(batch_size, 1, 128, 128)]

def get_init_inputs():
    return []

if __name__ == "__main__":
    batch_size = 1
    model = ModelNew()
    # y = model(torch.rand(batch_size, 1, 128, 128))
    traced = torch.fx.symbolic_trace(model)
    print(traced.code)
