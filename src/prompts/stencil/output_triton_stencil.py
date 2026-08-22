import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, None)
    tmp1 = tl.load(in_ptr0 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tl.store(in_out_ptr0 + x0, tmp3, None)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (1, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_2, (1,), (1,))
    assert_size_stride(primals_3, (1, 1, 10240, 10240), (10485760, 1048576,
        10240, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 
            1), padding=(3, 3), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (1, 1, 10240, 10240), (10485760, 1048576,
            10240, 1))
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(104857600)](buf1, primals_2, 
            10485760, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_2
    return buf1, primals_1, primals_3


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, stride=1, padding=3)
    
    def forward(self, input_0):
        primals_1 = self.conv1.weight
        primals_2 = self.conv1.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
