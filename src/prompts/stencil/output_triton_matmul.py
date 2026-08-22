architecture named Model with custom Triton kernels! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code!
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


@triton.jit
def triton_poi_fused_clone_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y0 = yindex % 8192
    y1 = yindex // 8192
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 8192 * x2 + 33554432 * y1), None,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 2048 * y3), tmp0, None)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg1_1, (4096, 2048), (2048, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 2048), (2048, 1), torch.float32)
        extern_kernels.mm(arg0_1, arg1_1, out=buf0)
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((1024, 2048, 2, 2), (8192, 4, 2, 1),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused_clone_0[grid(8388608, 2048)](buf0, buf1, 8388608,
            2048, XBLOCK=32, YBLOCK=32, num_warps=4, num_stages=1)
        del buf0
    return buf1,


class ModelNew(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, input_0, input_1):
        arg0_1 = input_0
        arg1_1 = input_1
        output = call([arg0_1, arg1_1])
        return output[0]
```