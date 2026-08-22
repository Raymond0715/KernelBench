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
def triton_poi_fused_clone_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 128
    xnumel = 8
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex
    x1 = xindex
    tmp0 = tl.load(in_ptr0 + (y0 + 128 * x1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + 8 * y0), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_add_1(in_out_ptr0, in_ptr0, ynumel, xnumel, YBLOCK: tl
    .constexpr, XBLOCK: tl.constexpr):
    ynumel = 64
    xnumel = 8
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_out_ptr0 + (y0 + 64 * x1), xmask & ymask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (y0 + 64 * x1), tmp2, xmask & ymask)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (1, 1, 128, 128), (16384, 16384, 128, 1))
    assert_size_stride(primals_2, (1, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_3, (1,), (1,))
    assert_size_stride(primals_4, (2, 49, 8), (392, 8, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 8), (8, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_clone_0[grid(128, 8)](primals_1, buf0, 128, 8,
            XBLOCK=8, YBLOCK=128, num_warps=4, num_stages=1)
        del primals_1
        buf1 = empty_strided_cuda((128, 49), (49, 1), torch.float32)
        extern_kernels.mm(buf0, reinterpret_tensor(primals_4, (8, 49), (1, 8
            ), 0), out=buf1)
        buf2 = empty_strided_cuda((128, 49), (49, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(buf0, (128, 8), (8, 1), 392),
            reinterpret_tensor(primals_4, (8, 49), (1, 8), 392), out=buf2)
        buf3 = reinterpret_tensor(buf1, (1, 1, 8, 8), (64, 64, 8, 1), 0)
        del buf1
        triton_poi_fused_add_1[grid(64, 8)](buf3, primals_3, 64, 8, XBLOCK=8,
            YBLOCK=32, num_warps=4, num_stages=1)
        del primals_3
        buf4 = reinterpret_tensor(buf2, (1, 1, 8, 8), (64, 64, 8, 1), 0)
        del buf2
        triton_poi_fused_add_1[grid(64, 8)](buf4, primals_4, 64, 8, XBLOCK=8,
            YBLOCK=32, num_warps=4, num_stages=1)
        del primals_4
    return buf3, buf4


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
    """
    def forward(self, x):
        x0 = x
        y0 = self.conv1(x0)
        y1 = self.conv1(x0)
        z = torch.cat((x0, y0), dim=1)
        output = torch.matmul(z, self.param_matrix)
        return output
    """

    def forward(self, input_0):
        primals_2 = self.conv1.weight
        primals_3 = self.conv1.bias
        primals_4 = self.param_matrix
        primals_1 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4])
        return output[0]
