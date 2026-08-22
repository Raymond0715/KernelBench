architecture named Model with custom Triton kernels! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! [/INST] 
```
import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_constant_pad_nd_0(in_ptr0, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1761280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 140 % 140
    x0 = xindex % 140
    x2 = xindex // 19600
    x4 = xindex
    tmp0 = -3 + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 136, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = -3 + x0
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (-390 + x0 + 136 * x1 + 18496 * x2), tmp10 &
        xmask, other=0.0)
    tl.store(out_ptr0 + x4, tmp11, xmask)


@triton.jit
def triton_poi_fused_add_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK:
    tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + x2, None)
    tmp1 = tl.load(in_out_ptr0 + x2, None)
    tmp3 = tl.load(in_ptr1 + x0, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + x2, tmp4, None)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (1, 1, 10240, 10240), (10485760, 1048576,
        10240, 1))
    assert_size_stride(primals_2, (1, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_3, (1,), (1,))
    assert_size_stride(primals_4, (2, 49, 8), (392, 8, 1))
    assert_size_stride(primals_5, (1, 1, 7, 7), (49, 49, 7, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 1, 140, 140), (19600, 19600, 140, 1),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_0[grid(1761280)](primals_1, buf0, 
            1761280, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_1
        buf1 = extern_kernels.convolution(buf0, primals_2, stride=(1, 1),
            padding=(3, 3), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (1, 1, 140, 140), (19600, 19600, 140, 1))
        buf2 = empty_strided_cuda((1, 8), (8, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(buf1, (1, 19600), (19600, 1), 0
            ), reinterpret_tensor(primals_4, (19600, 8), (1, 19600), 0), out
            =buf2)
        buf3 = empty_strided_cuda((1, 8), (8, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(buf1, (1, 19600), (19600, 1), 
            8), reinterpret_tensor(primals_4, (19600, 8), (1, 19600), 8),
            out=buf3)
        buf4 = reinterpret_tensor(buf2, (1, 1, 8), (8, 8, 1), 0)
        del buf2
        triton_poi_fused_add_1[grid(8)](buf4, buf3, primals_3, 8, XBLOCK=8,
            num_warps=1, num_stages=1)
        del buf3
        del primals_3
        buf5 = extern_kernels.convolution(buf4, primals_5, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (1, 1, 1, 1), (1, 1, 1, 1))
    return buf5, primals_2, primals_5, buf0, reinterpret_tensor(buf1, (1, 
        19600), (19600, 1), 0), reinterpret_tensor(buf1, (1, 19600), (19600,
        1), 8), buf4, reinterpret_tensor(primals_4, (8, 19600), (19600, 1), 8
        ), reinterpret_tensor(primals_4, (8, 19600), (19600, 1), 0)


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
    

    def forward(self, input_0):
        primals_2 = self.conv1.weight
        primals_3 = self.conv1.bias
        primals_4 = self.param_matrix
        primals_5 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]
```